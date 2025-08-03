import numpy as np
import torch
import os
import trimesh
import OpenEXR
import Imath
import cv2
from copy import deepcopy
from articulated_object import ArticulatedObject

class ArticulatedObjectRenderer:
	def __init__(
			self, model_id, category, blender_root, camera_intr, scale=1.0):
		
		# arguments
		self.model_id = str(model_id)
		self.category = category
		self.blender_root = blender_root
		self.scale = scale

		# camera intrinsic
		self.image_size = np.array(camera_intr.image_size)
		self.fx = (camera_intr.intrinsic[0] + camera_intr.intrinsic[1]) / 2
		self.fy = (camera_intr.intrinsic[0] + camera_intr.intrinsic[1]) / 2
		self.cx = self.image_size[1] / 2.0
		self.cy = self.image_size[0] / 2.0
		self.intrinsic_matrix = np.array([
			[self.fx, 0.0, self.cx],
			[0.0, self.fy, self.cy],
			[0.0, 0.0, 1.0]
		])
		# self.intrinsic_matrix = np.array([
		# 	[camera_intr.intrinsic[0], 0.0, camera_intr.intrinsic[2]],
		# 	[0.0, camera_intr.intrinsic[1], camera_intr.intrinsic[3]],
		# 	[0.0, 0.0, 1.0]
		# ])

		# get object
		self.articulated_object = ArticulatedObject(model_id, scale=scale)

	def generate_mobility_dataset(
			self, camera_poses, steps=3, offset=0.0, 
			joint_indices=None, joint_custom_limits=None, shadow_on=False,
			mode='sequential', split_list=[], view_name=''):

		# save directory
		if 'train' in split_list:
			dir_save_train = os.path.join(
				'datasets',
				'partnet_mobility_blender',
				self.category,
				self.model_id,
				f'{mode}_steps_{steps}_{view_name}')
			os.makedirs(dir_save_train, exist_ok=True)
		if 'test' in split_list:
			dir_save_test = os.path.join(
				'datasets',
				'partnet_mobility_blender_eval',
				self.category,
				self.model_id,
				f'{mode}_steps_{steps}_{view_name}')
			os.makedirs(dir_save_test, exist_ok=True)

		# set valid joint joint limits
		if (joint_indices is None) and (joint_custom_limits is None):
			joint_limits = torch.stack([values['limit'] 
				for _, values in self.articulated_object.S_screws.items()])
			joint_limits[:, 0] += offset
			joint_limits[:, 1] -= offset
		elif (joint_indices is not None) and (joint_custom_limits is None):
			joint_limits = torch.stack([values['limit']
				if i in joint_indices else torch.tensor([values['limit'][0], values['limit'][0]])
				for i, (_, values) in enumerate(self.articulated_object.S_screws.items())
			])
		elif (joint_indices is None) and (joint_custom_limits is not None):
			joint_limits = torch.tensor([joint_custom_limits])
		elif (joint_indices is not None) and (joint_custom_limits is not None):
			raise NotImplementedError

		# save screws and joint limits
		if 'test' in split_list:
			if (joint_indices is None) and (joint_custom_limits is None):
				S_screws_modified = deepcopy(self.articulated_object.S_screws)
			elif (joint_indices is not None) and (joint_custom_limits is None):
				S_screws_modified = {
					key: value
					for i, (key, value) in enumerate(self.articulated_object.S_screws.items())
					if i in joint_indices
				}
			elif (joint_indices is None) and (joint_custom_limits is not None):
				S_screws_modified = deepcopy(self.articulated_object.S_screws)
				for i, (keys, values) in enumerate(S_screws_modified.items()):
					values['limit'] = joint_limits[0]
			elif (joint_indices is not None) and (joint_custom_limits is not None):
				raise NotImplementedError
			np.save(
				os.path.join(dir_save_test, 'screws.npy'), S_screws_modified)

		# interpolates
		if mode == 'sequential': # simple linear interpolates
			thetas = torch.linspace(
				0, 1, steps, 
				device=joint_limits.device, 
				dtype=joint_limits.dtype
			).view(-1, *[1]*joint_limits[:, 0].dim())
			thetas = joint_limits[:, 0] + thetas * (joint_limits[:, 1] - joint_limits[:, 0])
		elif mode == 'random': # random sample
			thetas = torch.rand(steps, joint_limits.shape[0])
			thetas = joint_limits[:, 0] + thetas * (joint_limits[:, 1] - joint_limits[:, 0])

		# generate dataset
		for split in split_list:
			if split == 'train':
				thetas_ = thetas
			elif split == 'test':
				thetas_1 = torch.cat(
					[thetas[:1, :], thetas[1:, :].repeat_interleave(2, dim=0)])
				thetas_2 = torch.cat(
					[thetas[:-1, :].repeat_interleave(2, dim=0), thetas[-1:, :]])
				thetas_ = (thetas_1 + thetas_2) / 2

			for i, theta in enumerate(thetas_):
				if split == 'train':
					self.dir_save = dir_save_train
					folder_name = f'{i}'
				elif split == 'test':
					self.dir_save = dir_save_test
					folder_name = f'{i/2:.1f}'
				self.generate(
					camera_poses, theta, split=split, 
					folder_name=folder_name, shadow_on=shadow_on)

	def generate(
			self, camera_poses, theta, 
			split='train', folder_name='temp', shadow_on=False, n_pc=2048):

		# make folder and save intrinsic
		os.makedirs(os.path.join(self.dir_save, folder_name), exist_ok=True)
		save_intrinsic_name = os.path.join(self.dir_save, folder_name, 'intrinsic.npy')
		np.save(save_intrinsic_name, self.intrinsic_matrix)

		# save theta
		if split == 'test':
			np.save(os.path.join(self.dir_save, folder_name, 'theta.npy'), theta)

		# update object
		meshes, link_types = self.articulated_object.update_object(
			theta, return_link_type=True)
 
		# get point cloud
		if split == 'test':
			movable_idx = 1
			for meshes_part, link_type in zip(meshes, link_types):

				# process part mesh
				meshes_list = []
				for mesh in meshes_part:
					if isinstance(mesh, trimesh.Scene):
						mesh = mesh.dump()
						meshes_list.extend(mesh)
					else:
						meshes_list.extend([mesh])
				combined_mesh = trimesh.util.concatenate(meshes_list)

				# sample point clouds
				points, _ = trimesh.sample.sample_surface(combined_mesh, count=n_pc)

				# save as np file
				if link_type == 'static':
					save_part_pc_name = os.path.join(
						self.dir_save, folder_name, f'point_cloud_static.npy')
				elif link_type == 'movable':
					save_part_pc_name = os.path.join(
						self.dir_save, folder_name, f'point_cloud_movable_{movable_idx}.npy')
					movable_idx += 1
				np.save(save_part_pc_name, np.array(points))

		# process for combine mesh
		meshes_list = []
		meshes_flat = []
		for mesh in meshes:
			meshes_flat.extend(mesh)
		for mesh in meshes_flat:
			if isinstance(mesh, trimesh.Scene):
				mesh = mesh.dump()
				meshes_list.extend(mesh)
			else:
				meshes_list.extend([mesh])

		# for exporting mesh
		combined_mesh = trimesh.util.concatenate(meshes_list)

		# save whole point cloud
		if split == 'test':
			points, _ = trimesh.sample.sample_surface(combined_mesh, count=n_pc)
			save_whole_pc_name = os.path.join(
				self.dir_save, folder_name, f'point_cloud_whole.npy')
			np.save(save_whole_pc_name, np.array(points))

		# save mesh
		obj_data, texture_data = trimesh.exchange.obj.export_obj(
			combined_mesh, return_texture=True, include_normals=True)
		with open(os.path.join(self.dir_save, folder_name, 'mesh.obj'), "w") as f:
			f.write(obj_data)
			f.close()
		with open(os.path.join(self.dir_save, folder_name, 'material.mtl'), "wb") as f:
			f.write(texture_data['material.mtl'])
			f.close()
		with open(os.path.join(self.dir_save, folder_name, 'material_0.png'), "wb") as f:
			f.write(texture_data['material_0.png'])
			f.close()     

		# render
		for i, pose in enumerate(camera_poses):

			# save name
			save_name = f'image_{i:03d}'
			save_extrinsic_name = os.path.join(self.dir_save, folder_name, 'extrinsics', f'{save_name}.npy')
			save_camera_pose_name = os.path.join(self.dir_save, folder_name, 'camera_pose', f'{save_name}.npy')

			# subfolders
			if not os.path.exists(os.path.join(self.dir_save, folder_name, 'extrinsics')):
				os.makedirs(os.path.join(self.dir_save, folder_name, 'extrinsics'))
			if not os.path.exists(os.path.join(self.dir_save, folder_name, 'camera_pose')):
				os.makedirs(os.path.join(self.dir_save, folder_name, 'camera_pose'))

			# save extrinsic matrix
			rotx180 = np.array([
				[1, 0, 0, 0],
				[0, -1, 0, 0],
				[0, 0, -1, 0],
				[0, 0, 0, 1]
			])
			camera_pose = pose @ rotx180
			np.save(save_camera_pose_name, camera_pose)
			extrinsic = np.linalg.inv(pose)
			np.save(save_extrinsic_name, extrinsic)

			# run blender
			render_cmd = (
				f'{self.blender_root} -b -P articulated_object/render.py --'
				+ f' --dir_save {os.path.join(self.dir_save, folder_name)}'
				+ f' --camera_index {i}'
				+ f' --image_size {self.image_size[0]} {self.image_size[1]}'
				+ f' --intrinsics {self.fx} {self.fy} {self.cx} {self.cy}'
			)
			if shadow_on:
				render_cmd += f' --shadow_on'
			render_cmd += f' >> tmp.out'
			os.system(render_cmd)

			# load depth exr file
			depth_exr = OpenEXR.InputFile(
				os.path.join(self.dir_save, folder_name, 'depths_exr', f'depth_1_{i:03d}.exr'))

			# get the header to determine dimensions
			header = depth_exr.header()
			dw = header['displayWindow']
			width = dw.max.x - dw.min.x + 1
			height = dw.max.y - dw.min.y + 1

			# Define pixel type as 32-bit float
			pt = Imath.PixelType(Imath.PixelType.FLOAT)

			# read and convert to NumPy
			channels = ['B', 'G', 'R']
			depth_np = np.zeros((height, width, 3), dtype=np.float32)
			for j, c in enumerate(channels):
				channel_data = depth_exr.channel(c, pt)
				depth_np[:, :, j] = np.frombuffer(channel_data, dtype=np.float32).reshape((height, width))
			depth_np = np.mean(depth_np, axis=2)
			depth_np[depth_np == 10.0] = 0.0
			depth_np = (depth_np * 1000).astype(np.uint16)
	
			# save as png
			if not os.path.exists(os.path.join(self.dir_save, folder_name, 'depths')):
				os.makedirs(os.path.join(self.dir_save, folder_name, 'depths'))
			cv2.imwrite(
				os.path.join(self.dir_save, folder_name, 'depths', f'depth_{i:03d}.png'), 
				depth_np)	

if __name__ == '__main__':

	# object
	model_id = '10211'
	category = 'laptop'
	blender_root = None
	camera_intr = None

	# main
	articulated_object = ArticulatedObjectRenderer(
		model_id, category, blender_root, camera_intr)


	
