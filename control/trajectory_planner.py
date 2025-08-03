import torch
import os
import json
from argparse import ArgumentParser
from arguments import get_combined_args
from scene.gaussian_model import ScrewGaussianModel
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import time
from articulated_object.utils import exp_se3
from articulated_object.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from utils.sh_utils import SH2RGB
from robot.openchains_lib import Franka
from robot.openchains_torch import inverse_kinematics
import cv2

class TrajectoryPlanner():

	def __init__(self, args, center=[0.0, 0.0, 0.0], joint_angle=None):

		# center
		self.center = torch.tensor(center).float()

		# process args
		self.process_args(args)

		# initial joint angle
		if joint_angle is None:
			self.init_joint_angles = self.gaussians.get_joint_angles[-1].cpu()
		else:
			self.init_joint_angles = joint_angle.cpu()

		# for visualization
		self.get_visual_gaussians()
		self.get_visual_screws()

	def trajectory_planning(
			self, target_angle, num_points=100,
			return_tip_trajectory=False,
			offset_ee_angle=0.0, approach_distance=0.1):

		# get tip trajectory
		tip_trajectory = self.tip_trajectory_planning(
			self.init_joint_angles, target_angle, num_points=num_points,
			offset_ee_angle=offset_ee_angle
		)

		# end-effector to tip
		ee2tip = torch.eye(4)
		ee2tip[:3, 3] = torch.tensor([0.0, 0.0, 0.1034 - 0.013])        
		
		# final ee poses
		initial_SE3 = tip_trajectory[0].clone()
		initial_SE3[:3, 3] = initial_SE3[:3, 3] - approach_distance * initial_SE3[:3, 2]
		tip_trajectory = torch.cat([initial_SE3.unsqueeze(0), tip_trajectory], dim=0)
		ee_poses = tip_trajectory @ torch.inverse(ee2tip).unsqueeze(0).repeat(len(tip_trajectory), 1, 1)

		# ik settings
		franka = Franka(device='cpu')
		max_iter = 2000
		step_size1 = 0.1
		step_size2 = 0.001
		# step_size2 = 0.0
		tolerance = 1e-2

		# initial joint angle
		initial_joint_state = [
			0.0301173169862714, 
			-1.4702106391932968, 
			0.027855688427362513, 
			-2.437557753144649, 
			0.14663284881434122, 
			2.308719465520647, 
			0.7012385825324389]        

		# solve ik
		joint_angles = torch.tensor(
			initial_joint_state).unsqueeze(0).repeat(len(ee_poses), 1).to(ee_poses)
		out_joint_angles, dict_infos = inverse_kinematics(
			joint_angles, 
			ee_poses,
			franka,
			max_iter=max_iter,
			step_size1=step_size1,
			step_size2=step_size2,
			tolerance=tolerance,
			device='cpu')

		# check success
		successes = (dict_infos['final_error'] < tolerance) * dict_infos['joint limit check']
		print((dict_infos['final_error'] < tolerance))
		print(dict_infos['joint limit check'])
		print(f'[1] The number of ik success poses is {int(torch.sum(successes).item())}')

		# joint angle
		out_joint_angles = out_joint_angles[successes]

		if return_tip_trajectory:
			return out_joint_angles, tip_trajectory
		else:
			return out_joint_angles

	def tip_trajectory_planning(
			self, current_angle, target_angle,
			num_points=100,
			offset_revol=torch.pi/18, offset_pris=0.04,
			offset_ee_angle=0.0):

		# get base center
		base_label = (
			(torch.argmax(self.part_indices, dim=1) == 0)
			* (self.opacities > 0.5).squeeze(-1)
		)
		base_center = torch.mean(self.xyzs[base_label], dim=0)

		# get main screw
		i = torch.argmax(self.screw_confs)
		screw = self.screws[i]

		# screw type and offset
		eps = 1e-5
		offset_ee_angle = torch.tensor(offset_ee_angle)
		if torch.norm(screw[:3]) < eps:
			screw_type = 'pris'
			offset = offset_pris  
		else:
			screw_type = 'revol'
			offset = offset_revol

		# get part gaussian centers
		part_label = (
			(torch.argmax(self.part_indices, dim=1) == i + 1)
			* (self.opacities > 0.5).squeeze(-1)
		)
		centers = self.xyzs[part_label]
		n_centers = centers.shape[0]

		# compute initial point
		if screw_type == 'revol':
			
			# axis
			w = screw[:3]
			q = self._screws[i, 3:]
			
			# get distances
			dists = torch.cross(
				centers - q.unsqueeze(0).repeat(n_centers, 1),
				w.unsqueeze(0).repeat(n_centers, 1)
			)
			dists = torch.norm(dists, dim=1)

			# initial point sets
			initial_point_sets = centers[
				torch.argsort(dists)[int(0.95 * len(dists)) : int(len(dists))], :
			]
			n_init_points = initial_point_sets.shape[0]

			# get projections
			projections = torch.sum(
				(
					(initial_point_sets - q.unsqueeze(0).repeat(n_init_points, 1))
					* w.unsqueeze(0).repeat(n_init_points, 1)
				), dim=1
			)

			# initial point
			initial_point = initial_point_sets[
				torch.argsort(projections)[int(0.5 * len(projections))], :
			]

		elif screw_type == 'pris':

			# axis
			w = screw[3:]
			
			# get projections
			projections = torch.sum(
				(
					(centers - base_center.unsqueeze(0).repeat(n_centers, 1))
					* w.unsqueeze(0).repeat(n_centers, 1)
				), dim=1
			)
			projections = projections * torch.sign(projections)

			# initial point sets
			initial_point_sets = centers[
				torch.argsort(projections)[int(0.8 * len(projections)) : int(1.0 * len(projections))], :
			]
			initial_points_center = torch.mean(initial_point_sets, dim=0)
			n_init_points = initial_point_sets.shape[0]

			# get dists
			dists = torch.cross(
				initial_point_sets - initial_points_center.unsqueeze(0).repeat(n_init_points, 1),
				w.unsqueeze(0).repeat(n_init_points, 1)
			)
			dists = torch.norm(dists, dim=1)

			# initial point
			initial_point = initial_point_sets[
				torch.argsort(dists)[int(0.0 * len(dists))], :
			]

		# generate arc points
		thetas = torch.linspace(0, 1, num_points).to(self.init_joint_angles)
		thetas = (
			(1 - thetas) * (current_angle + offset * torch.sign(current_angle - target_angle))
			+ thetas * target_angle
		)
		screw_thetas = thetas.unsqueeze(-1) @ screw.unsqueeze(0) # [N, 6]
		screw_transforms = exp_se3(screw_thetas) # [N, 4, 4]
		screw_rotations = screw_transforms[:, :3, :3] # [N, 3, 3]
		screw_translations = screw_transforms[:, :3, 3] # [N, 3]
		trajectory = (
			screw_rotations @ initial_point.unsqueeze(0).repeat(num_points, 1).unsqueeze(-1)
			+ screw_translations.unsqueeze(-1)
		).squeeze(-1)
		
		# translate to center
		trajectory = trajectory + self.center.unsqueeze(0)

		# tip SE3s
		if screw_type == 'revol':
			
			# # make orientation (z-direction)
			# z_axes = trajectory[1:] - trajectory[:-1]
			# z_axes = z_axes / torch.norm(z_axes, dim=1, keepdim=True)
			# y_axes = torch.cross(z_axes, torch.tensor([[0.0, 0.0, 1.0]]).repeat(len(z_axes), 1))
			# x_axes = torch.cross(y_axes, z_axes)
			# SO3s = torch.stack([x_axes, y_axes, z_axes], dim=-1)
			# SO3s = SO3s @ offset_SO3.unsqueeze(0).repeat(len(SO3s), 1, 1)

			# make orientation (constant)
			y_axis = w * torch.sign(torch.sum(w * torch.tensor([0.0, -1.0, 0.0])))
			z_axis = trajectory[1] - trajectory[0]
			z_axis[2] = 0
			z_axis = z_axis / torch.norm(z_axis) * torch.sign(torch.sum(z_axis * torch.tensor([1.0, 0.0, 0.0])))
			x_axis = torch.cross(y_axis, z_axis)

			# offset
			offset_ee_angle = offset_ee_angle
			offset_SO3 = torch.tensor([
				[torch.cos(offset_ee_angle), 0.0, torch.sin(offset_ee_angle)],
				[0.0, 1.0, 0.0],
				[-torch.sin(offset_ee_angle), 0.0, torch.cos(offset_ee_angle)],
			])

			SO3s = torch.stack([x_axis, y_axis, z_axis], dim=-1).unsqueeze(0).repeat(len(trajectory), 1, 1)
			SO3s = SO3s @ offset_SO3.unsqueeze(0).repeat(len(SO3s), 1, 1)
			
			# make SE3s
			SE3s = torch.eye(4).unsqueeze(0).repeat(len(trajectory), 1, 1)
			SE3s[:, :3, :3] = SO3s
			SE3s[:, :3, 3] = trajectory

		elif screw_type == 'pris':

			# offset
			offset_SO3 = torch.tensor([
				[torch.cos(offset_ee_angle), 0.0, torch.sin(offset_ee_angle)],
				[0.0, 1.0, 0.0],
				[-torch.sin(offset_ee_angle), 0.0, torch.cos(offset_ee_angle)],
			])

			# make orientation (constant)
			y_axis = (
				torch.cross(w, torch.tensor([0.0, 0.0, 1.0]))
				* torch.sign(torch.sum(torch.cross(w, torch.tensor([0.0, 0.0, 1.0])) * torch.tensor([0.0, -1.0, 0.0])))
			)
			z_axis = w
			x_axis = torch.cross(y_axis, z_axis)

			SO3s = torch.stack([x_axis, y_axis, z_axis], dim=-1).unsqueeze(0).repeat(len(trajectory), 1, 1)
			SO3s = SO3s @ offset_SO3.unsqueeze(0).repeat(len(SO3s), 1, 1)

			# make SE3s
			SE3s = torch.eye(4).unsqueeze(0).repeat(len(trajectory), 1, 1)
			SE3s[:, :3, :3] = SO3s
			SE3s[:, :3, 3] = trajectory

		return SE3s

	def visualization(self, trajectory=None, tip_trajectory=None):

		# visualizer
		vis = o3d.visualization.Visualizer()
		vis.create_window(width=1280, height=720)

		# # robot initial pose
		# franka = Franka(
		# 	device='cpu', 
		# 	load_links=True, load_d435=True, load_gripper=True)
		# initial_joint_state = torch.tensor([
		# 	0.0301173169862714, 
		# 	-1.4702106391932968, 
		# 	0.027855688427362513, 
		# 	-2.437557753144649, 
		# 	0.14663284881434122, 
		# 	2.308719465520647, 
		# 	0.7012385825324389])
		# robot_mesh = franka.get_robot_mesh(initial_joint_state)
		# vis.add_geometry(robot_mesh)	

		# screw transforms
		screw_thetas = self.screws * self.init_joint_angles.unsqueeze(1)
		screw_transforms = exp_se3(screw_thetas).detach().numpy()

		# add gaussians
		for i, (segmented_ellipsoids, segmented_colors, segmented_opacities) in enumerate(zip(self.gaussian_ellipsoids, self.gaussian_colors, self.gaussian_opacities)):
			ellipsoids = o3d.geometry.TriangleMesh()
			for ellipsoid, color, opacity in zip(segmented_ellipsoids, segmented_colors, segmented_opacities):

				# deepcopy
				_ellipsoid = deepcopy(ellipsoid)

				# transform
				if i >= 1:
					_ellipsoid.transform(screw_transforms[i-1])

				# translate to center
				_ellipsoid.translate(self.center)

				# color
				_ellipsoid.paint_uniform_color(torch.clip(color, 0, 1))

				# append
				ellipsoids += _ellipsoid

			# add
			vis.add_geometry(ellipsoids)

		# add screws
		for i, (screw_line, screw_point, screw_color) in enumerate(zip(self.screw_lines, self.screw_points, self.screw_colors)):
			screw_line.paint_uniform_color(screw_color)
			screw_point.paint_uniform_color(screw_color)

			# add
			vis.add_geometry(screw_line)
			vis.add_geometry(screw_point)

		# add trajectory
		if tip_trajectory is not None:
			for SE3 in tip_trajectory:
				frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)
				frame.transform(SE3.cpu().detach().numpy())
				vis.add_geometry(frame)

		# camera view
		if os.path.exists("o3d_camera.json"):
			camera = o3d.io.read_pinhole_camera_parameters("o3d_camera.json")
			ctr = vis.get_view_control()
			ctr.convert_from_pinhole_camera_parameters(camera, True)

		if trajectory is not None:

			# declare robot
			franka = Franka(
				device='cpu', 
				load_links=True, load_d435=True, load_gripper=True)
			
			# add meshes
			for i, joint_angle in enumerate(trajectory):
			
				if i == 0:
					robot_mesh = franka.get_robot_mesh(joint_angle.detach())
					vis.add_geometry(robot_mesh)
				else:
					vis.remove_geometry(robot_mesh)
					robot_mesh = franka.get_robot_mesh(joint_angle.detach())
					vis.add_geometry(robot_mesh)
				vis.poll_events()
				vis.update_renderer()
		
		# run
		vis.run()

		# save recent camera view
		view_ctl = vis.get_view_control()
		param = view_ctl.convert_to_pinhole_camera_parameters()
		o3d.io.write_pinhole_camera_parameters("o3d_camera.json", param)

		# destroy window
		vis.destroy_window()

	def make_video(self, trajectory=None, tip_trajectory=None):
		
		# figure settings
		image_size = [960, 960]
		resolution = 1000
		voxel_size = 0.005
		voxel_size_scale = 0.7
		light_dir = (0.3, -0.3, -0.9)
		bbox_thickness = 0.004
		gray_color = np.array([0.7, 0.7, 0.7])

		# define ground plane
		a = 200.0
		plane = o3d.geometry.TriangleMesh.create_box(width=a, depth=0.05, height=a)
		plane.paint_uniform_color([1.0, 1.0, 1.0])
		plane.translate([-a/2, -a/2, -1.0])
		plane.compute_vertex_normals()
		mat_plane = rendering.MaterialRecord()
		mat_plane.shader = 'defaultLit'
		mat_plane.base_color = [1.0, 1.0, 1.0, 4.0]
		mat_gripper = rendering.MaterialRecord()
		mat_gripper.shader = 'defaultLitTransparency'
		mat_gripper.base_color = [1.0, 1.0, 1.0, 0.8]
		mat_before = rendering.MaterialRecord()
		mat_before.shader = 'defaultLitTransparency'
		mat_before.base_color = [1.0, 1.0, 1.0, 0.4]

		# object material
		mat = rendering.MaterialRecord()
		mat.shader = 'defaultLit'
		mat.base_color = [1.0, 1.0, 1.0, 0.9]

		# draw voxel
		image_size = [1280, 960]
		widget = o3d.visualization.rendering.OffscreenRenderer(
			image_size[0], image_size[1])

		# rendering camera info
		workspace_origin = np.array([0.3, 0.0, 0.3])
		camera_position = np.array(
			[0.3, 0.9, 0.9])
		camera_lookat = workspace_origin

		# camera viewpoint
		widget.scene.camera.look_at(camera_lookat, camera_position, [0,0,1])

		# other settings
		widget.scene.set_lighting(
			widget.scene.LightingProfile.DARK_SHADOWS, 
			(-0.3, 0.3, -0.9))
		widget.scene.set_background(
			[1.0, 1.0, 1.0, 4.0], 
			image=None)

		# screw transforms
		screw_thetas = self.screws * self.init_joint_angles.unsqueeze(1)
		screw_transforms = exp_se3(screw_thetas).detach().numpy()

		# add gaussians
		for i, (segmented_ellipsoids, segmented_colors, segmented_opacities) in enumerate(zip(self.gaussian_ellipsoids, self.gaussian_colors, self.gaussian_opacities)):
			ellipsoids = o3d.geometry.TriangleMesh()
			for ellipsoid, color, opacity in zip(segmented_ellipsoids, segmented_colors, segmented_opacities):

				# deepcopy
				_ellipsoid = deepcopy(ellipsoid)

				# transform
				if i >= 1:
					_ellipsoid.transform(screw_transforms[i-1])

				# translate to center
				_ellipsoid.translate(self.center)

				# color
				_ellipsoid.paint_uniform_color(torch.clip(color, 0, 1))

				# append
				ellipsoids += _ellipsoid

			# add
			widget.scene.add_geometry(f'ellipsoids{i}', ellipsoids, mat)

		# add screws
		for i, (screw_line, screw_point, screw_color) in enumerate(zip(self.screw_lines, self.screw_points, self.screw_colors)):
			screw_line.paint_uniform_color(screw_color)
			screw_point.paint_uniform_color(screw_color)

		# add trajectory
		if tip_trajectory is not None:
			for i, SE3 in enumerate(tip_trajectory):
				if i % 12 != 0:
					continue
				frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)
				frame.transform(SE3.cpu().detach().numpy())
				widget.scene.add_geometry(f'frame{i}', frame, mat)

		# image paths
		os.makedirs("temp_figures5", exist_ok=True)
		image_paths = []

		# add trajectory
		if trajectory is not None:

			# declare robot
			franka = Franka(
				device='cpu', 
				load_links=True, load_d435=True, load_gripper=True)

			# add meshes
			for i, joint_angle in enumerate(trajectory):
					
				print(i)
			
				if i == 0:
					robot_mesh = franka.get_robot_mesh(joint_angle.detach())
					widget.scene.add_geometry('robot_mesh', robot_mesh, mat)
				else:
					widget.scene.remove_geometry('robot_mesh')
					robot_mesh = franka.get_robot_mesh(joint_angle.detach())
					widget.scene.add_geometry('robot_mesh', robot_mesh, mat)

				# render RGB image
				img = widget.render_to_image()
				image_path = f"temp_figures5/temp{i}.png"
				o3d.io.write_image(image_path, img)
				image_paths.append(image_path)

		# make video 
		output_video = 'output_video5.mp4'
		fps = 30

		# Read first image to get size
		frame = cv2.imread(image_paths[0])
		height, width, _ = frame.shape

		# Define the video writer
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
		video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

		# for white background
		def composite_on_white(img):
			if img.shape[2] == 4:
				alpha = img[:, :, 3] / 255.0
				rgb = img[:, :, :3]
				white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
				blended = (alpha[..., None] * rgb + (1 - alpha[..., None]) * white_bg).astype(np.uint8)
				return blended
			else:
				return img

		# Write each frame
		for img_path in image_paths:
			img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
			img_rgb = composite_on_white(img)
			video.write(img_rgb)	

	def get_visual_gaussians(self):

		# process gaussians
		self.gaussian_ellipsoids = [[] for _ in range(self.part_indices.shape[1])]
		self.gaussian_colors = [[] for _ in range(self.part_indices.shape[1])]
		self.gaussian_opacities = [[] for _ in range(self.part_indices.shape[1])]
		for i in range(len(self.xyzs)):
			scale = self.scalings[i]  
			SO3 = self.SO3s[i] 
			position = self.xyzs[i]
			# color = features[i, 0, :]
			color = SH2RGB(self.features[i, 0, :])
			opacity = self.opacities[i]
			part_index = np.argmax(self.part_indices[i])

			# Create a unit sphere mesh (default sphere in Open3D)
			ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=4)
			ellipsoid.compute_vertex_normals()

			# Scale to Gaussian covariance
			scaling_matrix = np.diag(scale)  # Convert to (3,3) diagonal scaling matrix
			transform_matrix = np.eye(4)  # 4x4 transformation matrix
			transform_matrix[:3, :3] = SO3 @ scaling_matrix  # Apply rotation & scaling
			transform_matrix[:3, 3] = position  # Set translation

			ellipsoid.transform(transform_matrix)

			if opacity > 0.3:
				self.gaussian_ellipsoids[part_index].append(ellipsoid)
				self.gaussian_colors[part_index].append(color)
				self.gaussian_opacities[part_index].append(opacity)

	def get_visual_screws(self):

		# draw screws
		self.screw_lines = []
		self.screw_points = []
		self.screw_colors = []
		for j in range(len(self.screws)):

			# initialize
			w = self.screws[j, :3]
			v = self.screws[j, 3:]
			q = self._screws[j, 3:]
			conf = self.screw_confs[j]
			linelen = 0.5

			# revolute
			if np.linalg.norm(w) > 1e-6:
				start = q + self.center - linelen * w
				end = q + self.center + linelen * w
				color = [conf, 0, 0]

				# line segment
				line_set = o3d.geometry.LineSet()
				points = [start, end]
				lines = [[0, 1]]
				line_set.points = o3d.utility.Vector3dVector(points)
				line_set.lines = o3d.utility.Vector2iVector(lines)

				# screw point
				sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=10)
				sphere.translate(q + self.center)

			# prismatic
			else:
				start = self.center -linelen * v
				end = self.center + linelen * v
				color = [0, 0, conf]

				# line segment
				line_set = o3d.geometry.LineSet()
				points = [start, end]
				lines = [[0, 1]]
				line_set.points = o3d.utility.Vector3dVector(points)
				line_set.lines = o3d.utility.Vector2iVector(lines)

				# screw point
				sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=10)
			
			self.screw_lines.append(line_set)
			self.screw_points.append(sphere)
			self.screw_colors.append(color)

	def process_args(self, args):

		# process args
		source_path = args.source_path
		model_path = args.model_path
		images = args.images
		depths = args.depths
		white_background = args.white_background
		sh_degree = args.sh_degree

		# initialize gaussian
		checkpoint_name_list = [file for file in os.listdir(model_path) if file.endswith('.pth')]
		sorted_checkpoints_names = sorted(checkpoint_name_list, key=lambda x: int(x.replace('chkpnt', '').replace('.pth', '')))
		sorted_iters = [int(x.replace('chkpnt', '').replace('.pth', '')) for x in sorted_checkpoints_names]
		checkpoint_name = sorted_checkpoints_names[-1]

		self.gaussians = ScrewGaussianModel(sh_degree)

		# load gaussians
		checkpoint = os.path.join(model_path, checkpoint_name)
		# (model_params, first_iter) = torch.load(checkpoint)
		(model_params, first_iter) = torch.load(checkpoint, weights_only=False)
		self.gaussians.restore(model_params)

		# get gaussians
		self.scalings = self.gaussians.get_scaling.cpu().detach()
		rotations = self.gaussians.get_rotation.cpu().detach()
		self.SO3s = quaternion_to_matrix(rotations).detach()
		self.xyzs = self.gaussians.get_xyz.cpu().detach()
		self.features = self.gaussians.get_features.cpu().detach()
		self.opacities = self.gaussians.get_opacity.cpu().detach()
		self.part_indices = self.gaussians.get_part_indices.cpu().detach()
		
		# get screws
		self.screws = self.gaussians.get_screws.cpu().detach()
		self._screws = self.gaussians._screws.cpu().detach()
		self.screw_confs = self.gaussians.get_screw_confs.cpu().detach()
		
		# get joint angles
		joint_angles = self.gaussians.get_joint_angles
		lower_limit = torch.min(torch.stack(joint_angles), dim=0)[0]
		upper_limit = torch.max(torch.stack(joint_angles), dim=0)[0]
		joint_limits = torch.cat(
			(lower_limit.unsqueeze(-1), upper_limit.unsqueeze(-1)), dim=1
		).cpu().detach()

if __name__ == "__main__":

	# parser
	parser = ArgumentParser(description="Testing script parameters")
	parser.add_argument("--model_path", type=str)
	args = get_combined_args(parser)

	# trajectory planner
	trajectory_planner = TrajectoryPlanner(args)