from argparse import ArgumentParser
import os
import time
import cv2
import threading
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
# from lang_sam import LangSAM

from arguments import get_combined_args
from articulated_object.get_camera_poses import get_camera_poses
from robot.openchains_lib import Franka
from robot.openchains_torch import inverse_kinematics
from control.communicator_server import Listener
from control.image_to_joints import Image2JointAngles
from control.text_to_joints import Text2JointAngles
from control.trajectory_planner import TrajectoryPlanner


class Controller:
	def __init__(self, cfg):

		# object class
		self.object_class = cfg.object_class
		self.max_art_indices = cfg.max_art_indices
		self.task_list = cfg.task_list
		self.precollected = cfg.get('precollected', None)
		self.prerecognized = cfg.get('prerecognized', None)
		self.offset_ee_angle = torch.pi * cfg.get('offset_ee_angle', -1/4)

		# defaults
		self.device = cfg.device
		self.device_for_ik = 'cpu'		

		# recognition setting
		self.parsimony_weight = cfg.get('parsimony_weight', 0.002)

		# real world setting
		self.ip = cfg.ip
		self.port = cfg.port

		# workspace center
		self.center = np.array(cfg.center)

		# camera settings
		self.get_intrinsics()
		self.get_camera_poses(cfg)
		ik_mode = cfg.get("ik_mode", False)
		if ik_mode:
			raise ValueError('the current mode is inverse kinematics mode')

		# observation settings
		if 'observation' in self.task_list:

			# data folder
			self.folder_name = os.path.join(
				'real_world',
				f'{self.object_class}_real',
				datetime.now().strftime('%Y%m%d-%H%M'),
				f'hand_steps_{self.max_art_indices}_partial_{len(self.camera_poses)}'
			)
			self.dataset_folder = os.path.join('datasets', self.folder_name)
			os.makedirs(self.dataset_folder, exist_ok=True)

		# segmentation setting
		self.mask_predictor = LangSAM()
		self.mask_predictor.device = self.device
		self.save_depth_imgs = cfg.get("save_depth_imgs", False)
		self.background_sam = cfg.get("background_sam", True)

		# for real-time segmentation inference
		if self.background_sam:
			self.initialize_img_list()
			self.finish_segmentation = False
			self.event = threading.Event()
			self.thread_sam = threading.Thread(
				target=self.realtime_segmentation, daemon=True)
			self.thread_sam.start()

	def recognition_and_control(self):

		#############################################################
		######################## OBSERVATION ########################
		#############################################################
		if 'observation' in self.task_list: 
			print('\033[95m' + 'Do observation? (y)' + '\033[0m')
			do_observation = input()
			if do_observation != 'y':
				return 0

			# request robot to observe
			self.listener = Listener(self.ip, self.port)
			data = self.listener.recv_vision()
			if not data == b'which_task':
				raise ValueError('invalid receved data; must be "which_task"')
			self.listener.send_grasp('observation')

			# observation
			for i in range(self.max_art_indices):
					
				print(
					f'''
					*************************************************
					************* RECOGNITION AND CONTROL ***********
					*************** ARTICULATION INDEX {i} ************
					*************************************************
					'''
				)

				# declare listener
				self.listener = Listener(self.ip, self.port)

				# make folder for i'th art index
				self.dataset_idx_folder = os.path.join(self.dataset_folder, str(i))
				os.makedirs(self.dataset_idx_folder, exist_ok=True)

				# save camera infos
				self.save_camera_infos(self.dataset_idx_folder)

				# observation
				self.initialize_img_list()
				self.finish_segmentation = False
				self.observation(folder=self.dataset_idx_folder)

				# close connection
				self.listener.close_connection()

			# make robot waiting for recognition
			self.listener = Listener(self.ip, self.port)
			data = self.listener.recv_vision()
			if not data == b'request_joint_angle':
				raise ValueError('invalid receved data; must be "request_joint_angle"')
			self.listener.send_grasp('observation_finished')

		#############################################################
		######################## RECOGNITION ########################
		#############################################################
		if 'recognition' in self.task_list:
			print('\033[95m' + 'Do recognition? (y)' + '\033[0m')
			do_recognition = input()
			if do_recognition != 'y':
				return 0
			
			if 'observation' in self.task_list:
				render_cmd = (
					'python train.py'
					+ f' -s {self.dataset_folder}'
					+ f' --parsimony_weight_init {self.parsimony_weight}'
					+ f' --parsimony_weight_final {self.parsimony_weight}'
				)
			else:
				if self.precollected is not None:
					render_cmd = (
						'python train.py'
						+ f' -s {self.precollected}'
						+ f' --parsimony_weight_init {self.parsimony_weight}'
						+ f' --parsimony_weight_final {self.parsimony_weight}'
					)
				else:
					raise ValueError('Without recognition, prerecognized model is needed.')                
			os.system(render_cmd)

		#############################################################
		######################## MANIPULATION #######################
		#############################################################
		if 'manipulation' in self.task_list:
			print('\033[96m' + 'Do manipulation? (y)' + '\033[0m')
			do_manipulation = input()
			if do_manipulation != 'y':
				return 0

			# load ScrewSplat
			if 'observation' in self.task_list:
				model_path = get_most_recent_subfolder(
					os.path.join('output', self.folder_name)
				)
			else:
				if 'recognition' in self.task_list:
					model_path = get_most_recent_subfolder(
						os.path.join('output', os.sep.join(self.precollected.split(os.sep)[1:]))
					)
				else:
					if self.prerecognized is not None:
						model_path = get_most_recent_subfolder(self.prerecognized)
					else:
						raise ValueError('Without recognition, prerecognized model is needed.')
			parser = ArgumentParser(description="Testing script parameters")
			parser.add_argument("--model_path", type=str, default=model_path)
			parser.add_argument('--object_class', type=str)
			parser.add_argument('--config', type=str)
			args = get_combined_args(parser)                
			
			# current joint angle
			latest_joint_angles = None
			while True:
				print('\033[96m' + 'Did you change the configuration of the object? (y/n)' + '\033[0m')
				value = input()

				# get current joint angle
				if value == 'y':

					# request robot to observe
					print('Estimate the joint angle.')

					# folder name
					if 'observation' in self.task_list:
						self.eval_folder = os.path.join(
							'datasets',
							'real_world_eval', 
							os.sep.join(self.dataset_folder.split(os.sep)[2:])
						)
					elif 'recognition' in self.task_list:
						self.eval_folder = os.path.join(
							'datasets',
							'real_world_eval', 
							os.sep.join(self.precollected.split(os.sep)[2:])
						)
					else:
						self.eval_folder = os.path.join(
							'datasets',
							'real_world_eval', 
							os.sep.join(self.prerecognized.split(os.sep)[2:])
						)

					# make folder for i'th art index
					dataset_eval_folder = os.path.join(
						self.eval_folder, str(0))
					os.makedirs(dataset_eval_folder, exist_ok=True)

					# reobserve or not
					while True:
						print('\033[96m' + 'Re-observation? (y/n)' + '\033[0m')
						value = input()

						# reobservation
						if value == 'y':
							
							# request to observe
							self.listener = Listener(self.ip, self.port)
							data = self.listener.recv_vision()
							if not data == b'which_task':
								raise ValueError('invalid receved data; must be "which_task"')
							self.listener.send_grasp('observation')
							self.listener.close_connection()

							# declare listener
							self.listener = Listener(self.ip, self.port)

							# save camera infos
							self.save_camera_infos(dataset_eval_folder)

							# observation
							self.initialize_img_list()
							self.finish_segmentation = False
							self.observation(folder=dataset_eval_folder)

							# close connection
							self.listener.close_connection()

							# make robot waiting for manipulation
							self.listener = Listener(self.ip, self.port)
							data = self.listener.recv_vision()
							if not data == b'request_joint_angle':
								raise ValueError('invalid receved data; must be "request_joint_angle"')
							self.listener.send_grasp('observation_finished')
							self.listener.close_connection()
							break

						elif value == 'n':
							print('Use the saved observations.')
							break
						else:
							print('invalid_keys. press y/n')

					# end thread
					self.end_thread(self.thread_sam)
					del self.mask_predictor

					# initialize image2joints model
					image_to_joints = Image2JointAngles(
						args, joint_angle=latest_joint_angles)

					# get observations
					images = image_to_joints.get_test_images(
						args, source_path=self.eval_folder)

					# joint angle optimization
					current_joint_angles = image_to_joints.bayes_optimize(images)
					
					# visualization
					image_to_joints.visualization(current_joint_angles, images)
					break

				# get latest joint angle
				elif value == 'n':
					print('Use the latest joint angle.')
					current_joint_angles = latest_joint_angles
					
					# end thread
					self.end_thread(self.thread_sam)
					del self.mask_predictor
					break
				else:
					print('invalid_keys. press y/n')

			# get target joint angle
			print('\033[96m' + 'Please enter the reference prompt.' + '\033[0m')
			ref_prompt = input()
			print('\033[96m' + 'Please enter the target prompt' + '\033[0m')
			tar_prompt = input()
			text_to_joints = Text2JointAngles(
				args, ref_prompt=ref_prompt, tar_prompt=tar_prompt,
				joint_angle=current_joint_angles)
			text_to_joints.get_CLIP_loss = text_to_joints.get_CLIP_dir_loss
			target_joint_angles = text_to_joints.bayes_optimize()

			# visualization
			text_to_joints.visualization(target_joint_angles)

			# trajectory planning
			trajectory_planner = TrajectoryPlanner(
				args, center=self.center,
				joint_angle=current_joint_angles)
			trajectory, tip_trajectory = trajectory_planner.trajectory_planning(
				target_joint_angles.cpu(), num_points=10, 
				offset_ee_angle=self.offset_ee_angle,
				return_tip_trajectory=True)
			data_trajectory = {
				'joint_angles': trajectory.detach().cpu().numpy().astype(np.float64),
			}

			# visualization
			trajectory_planner.visualization(
				trajectory=trajectory, tip_trajectory=tip_trajectory)

			# request robot to manipulate
			self.listener = Listener(self.ip, self.port)
			data = self.listener.recv_vision()
			if not data == b'which_task':
				raise ValueError('invalid receved data; must be "which_task"')
			self.listener.send_grasp('manipulation')
			self.listener.close_connection()

			# manipulate
			self.listener = Listener(self.ip, self.port)
			data = self.listener.recv_vision()
			if not data == b'request_joint_angle':
				raise ValueError('invalid receved data; must be "request_joint_angle"')
			self.listener.send_grasp(data_trajectory)
			self.listener.close_connection()

	#############################################################
	########################## FINISHED #########################
	#############################################################

		print(
			f'''
			*************************************************
			****************** FINISHED *********************
			*************************************************
			'''
		)

	#############################################################
	####################### OBSERVATION #########################
	#############################################################

	def initialize_img_list(self):
		self.rgb_raw_img_list = [None] * len(self.camera_poses)
		if self.save_depth_imgs:
			self.depth_img_list = [None] * len(self.camera_poses) 
		if self.background_sam:
			self.mask_img_list = [None] * len(self.camera_poses)
		self.segmented_indices = [False] * len(self.camera_poses)

	def observation(self, folder=None):

		# make folders
		os.makedirs(os.path.join(folder, 'raw_images'), exist_ok=True)
		os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
		if self.save_depth_imgs:
			os.makedirs(os.path.join(folder, 'depths'), exist_ok=True)

		# send joint angle
		print('getting observation data...')
		data = self.listener.recv_vision()
		if not data == b'request_joint_angle':
			raise ValueError('invalid receved data; must be "request_joint_angle"')
		self.listener.send_grasp(self.joint_angles)

		# receive vision data
		for i in range(len(self.joint_angles)):
			data = self.listener.recv_vision()
			if data == b'failed':
				self.listener.send_grasp('failed')
				return None, None
			rgb_raw_image = data[b'rgb_image']
			depth_image = data[b'depth_image']
			self.listener.send_grasp(f'{i+1}th_view_received')

			# save image
			save_rgb_name = os.path.join(
				folder, 'raw_images', f'image_{i:03d}.png')
			img = Image.fromarray(rgb_raw_image)
			img.save(save_rgb_name)

			# numpy to torch
			rgb_raw_image = torch.tensor(
				rgb_raw_image).float().permute(2, 0, 1)
			if self.save_depth_imgs:
				depth_image = torch.tensor(
					depth_image.astype(np.float32)).float()

			# append
			self.rgb_raw_img_list[i] = rgb_raw_image
			if self.save_depth_imgs:
				self.depth_img_list[i] = depth_image

		# stack
		if self.background_sam:
			self.finish_segmentation = False
			while not self.finish_segmentation:
				time.sleep(0.01)

			for i in range(len(self.joint_angles)):
				
				# save rgba image
				rgb_raw_image = self.rgb_raw_img_list[i]
				mask_image = self.mask_img_list[i]
				rgb_image = self.rgb2rgba(rgb_raw_image, mask_image)
				save_rgb_name = os.path.join(
					folder, 'images', f'image_{i:03d}.png')
				img = Image.fromarray(rgb_image, mode='RGBA')
				img.save(save_rgb_name)            

				# save depth image
				if self.save_depth_imgs:
					depth_image = self.depth_img_list[i]
					mask_image = self.mask_img_list[i]
					depth_image = (depth_image * mask_image).cpu().numpy().astype(np.uint16)
					save_depth_name = os.path.join(
						folder, 'depths', f'depth_{i:03d}.png')
					img = Image.fromarray(depth_image)
					img.save(save_depth_name)

	#############################################################
	####################### SEGMENTATION #########################
	#############################################################

	def rgb2rgba(self, rgb_raw, mask):

		# normalize mask to [0, 255] if not already
		if mask.max() == 1.0:
			mask = mask * 255.0

		# add alpha channel
		alpha = mask.unsqueeze(0)  # (1, H, W)
		rgba = torch.cat((rgb_raw, alpha), dim=0)  # (4, H, W)
		rgba = rgba.byte()

		# torch to numpy
		rgba_np = rgba.permute(1, 2, 0).cpu().numpy()

		return rgba_np

	def rgb2mask(self, imgs, text_prompt, num_aug=5):
		# get mask tensors from RGB 255 scale imgs
		
		# segmentation without jittering
		image_pil = Image.fromarray(
			imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
		)
		masks, _, _, _ = self.mask_predictor.predict(image_pil, text_prompt)
		masks = (masks.sum(dim=0) > 0).float()
		return masks

		# segmentation with jittering
		hue_trans = torch.linspace(-0.25, 0.25, num_aug-1)
		mask_jitt = []
		for _idx in range(num_aug):
			if _idx > 0:
				imgs_aug = T.functional.adjust_hue(imgs/255., hue_trans[_idx-1])
			else:
				imgs_aug = imgs
			mask_imgs = []
			for img in imgs_aug:
				image_pil = Image.fromarray(
					(img*255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
				)
				masks, _, _, _ = self.mask_predictor.predict(image_pil, text_prompt)
				masks = (masks.sum(dim=0) > 0).float()
				mask_imgs.append(masks)
			mask_imgs = torch.stack(mask_imgs).to(self.device)
			mask_jitt.append(mask_imgs)
		mask_jitt = torch.stack(mask_jitt) #  J x V x H x W
		mask_jitt = (mask_jitt.sum(dim=0) >= 0.99).float() # V x H x W
		return mask_jitt

	def realtime_segmentation(self):
		
		# run background
		while not self.event.is_set():
			
			# initialize
			all_segmented = False
			
			# run until all rgbs are segmented
			for i, mask_img in enumerate(self.mask_img_list):

				# mask image is empty
				if mask_img is None:

					# segmentation	
					if self.rgb_raw_img_list[i] is not None:
						start_time = time.time()
						result = self.rgb2mask(
							self.rgb_raw_img_list[i].unsqueeze(0), self.object_class)
						end_time = time.time()
						# self.mask_img_list[i] = result.squeeze(0)
						self.mask_img_list[i] = result.squeeze(0).cpu()
						if not self.segmented_indices[i]:
							self.segmented_indices[i] = True
							print(f'{i+1}th rgb image is segmented using SAM, shape: {self.mask_img_list[i].shape}, ellapsed time: {end_time - start_time}(s)')
					else:
						break

				# all images are segmented
				if i == len(self.mask_img_list) - 1:
					all_segmented = True

			# segmentation finished
			if all_segmented:
				self.finish_segmentation = True

	def end_thread(self, thread:threading.Thread):
		self.event.set()
		while thread.is_alive():
			time.sleep(0.1)
		self.event.clear()

	#############################################################
	######################### CAMERAS ###########################
	#############################################################

	def save_camera_infos(self, folder):
		
		# save intrinsic
		save_intrinsic_name = os.path.join(folder, 'intrinsic.npy')
		np.save(save_intrinsic_name, self.intrinsic_matrix)

		# save extrinsics
		os.makedirs(os.path.join(folder, 'extrinsics'), exist_ok=True)
		os.makedirs(os.path.join(folder, 'camera_pose'), exist_ok=True)
		for i, pose in enumerate(self.camera_poses):
			
			# save name
			save_extrinsic_name = os.path.join(
				folder, 'extrinsics', f'image_{i:03d}.npy')
			save_camera_pose_name = os.path.join(
				folder, 'camera_pose', f'image_{i:03d}.npy')

			# save extrinsic matrix
			pose_ = deepcopy(pose)
			pose_[:3, 3] = pose_[:3, 3] - self.center
			rotx180 = np.array([
				[1, 0, 0, 0],
				[0, -1, 0, 0],
				[0, 0, -1, 0],
				[0, 0, 0, 1]
			])
			camera_pose = pose_ @ rotx180
			np.save(save_camera_pose_name, camera_pose)
			extrinsic = np.linalg.inv(pose_)
			np.save(save_extrinsic_name, extrinsic)
			
	def get_intrinsics(self):
		
		# realsense intrinsic parameter
		self.intrinsic_matrix = np.array([
			[606.1148681640625, 0.0, 325.19329833984375],
			[0.0, 605.2857055664062, 246.53085327148438],
			[0.0, 0.0, 1.0]
		])

	def get_camera_poses(self, cfg):
		
		# camera settings
		num_phi = cfg.num_phi
		phi_range = cfg.phi_range
		num_theta = cfg.num_theta
		theta_range = cfg.theta_range
		radius = cfg.radius
		center = cfg.center

		# inverse kinematics settings
		max_iter = 2000
		step_size1 = 0.1
		step_size2 = 0.001
		tolerance = 1e-4
		initial_joint_state = [
			0.0301173169862714, 
			-1.4702106391932968, 
			0.027855688427362513, 
			-2.437557753144649, 
			0.14663284881434122, 
			2.308719465520647, 
			0.7012385825324389]

		# end-effector to camera
		ee2cam = torch.eye(4)
		ee2cam[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
		ee2cam[:3, 3] = torch.tensor([0.069, -0.0325, 0.01])

		# get camera poses
		camera_poses = get_camera_poses(
			num_phi=num_phi,
			phi_range=phi_range,
			num_theta=num_theta,
			theta_range=theta_range,
			radius=radius,
			cam_center=center
		)
		camera_poses = torch.tensor(camera_poses).float()		

		# inverse kinematics
		franka = Franka(device='cpu', load_links=True, load_d435=True)
		ee_poses = camera_poses @ torch.inverse(ee2cam).unsqueeze(0).repeat(len(camera_poses), 1, 1)

		# solve ik
		joint_angles = torch.tensor(
			initial_joint_state).unsqueeze(0).repeat(len(camera_poses), 1).to(ee_poses)
		joint_angles, dict_infos = inverse_kinematics(
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
		
		# get camera poses and joint angles
		self.camera_poses = camera_poses[successes]
		self.joint_angles = joint_angles[successes].numpy().astype(np.float64)
		print(f'The number of camera poses is {len(self.camera_poses)}')

#############################################################
######################### UTILS ###########################
#############################################################

def get_most_recent_subfolder(parent_folder):
	subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
				  if os.path.isdir(os.path.join(parent_folder, d))]
	if not subfolders:
		return None  # No subfolders exist

	# Sort subfolders by modification time (most recent last)
	subfolders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
	return subfolders[0]