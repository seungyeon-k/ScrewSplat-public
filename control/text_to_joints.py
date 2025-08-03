import torch
import os
from argparse import ArgumentParser
from arguments import get_combined_args
from gaussian_renderer import ScrewGaussianModel
import numpy as np
from gaussian_renderer import render_with_screw
from scene import ScrewGaussianModel
from PIL import Image
from scene.dataset_readers import CamerawithArticulationInfo
from typing import NamedTuple
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from utils.graphics_utils import getWorld2View2
import matplotlib.pyplot as plt
from copy import deepcopy
from skopt import gp_minimize
import imageio.v2 as imageio

class MiniCam(NamedTuple):
	FoVx: np.array
	FoVy: np.array
	image_width: int
	image_height: int
	cx: np.array
	cy: np.array
	art_idx: int
	world_view_transform: torch.tensor

class Text2JointAngles:

	def __init__(
			self, args, ref_prompt, tar_prompt, 
			joint_angle=None, joint_angle_idx=None):
		
		# device
		self.device = "cuda"

		# load CLIP module
		# clip_model = "openai/clip-vit-large-patch14"
		clip_model = "openai/clip-vit-base-patch32"
		self.load_modules(clip_model=clip_model)

		# process
		self.process_data(args)
	
		# initial joint angle
		if joint_angle is None:
			if joint_angle_idx is None:
				self.init_joint_angles = self.gaussians.get_joint_angles[-1]
			else:
				self.init_joint_angles = self.gaussians.get_joint_angles[joint_angle_idx]
		else:
			self.init_joint_angles = joint_angle.to(self.device)

		# get initial features
		with torch.no_grad():
			self.process_text_prompts(ref_prompt, tar_prompt)
			self.init_image_feature = self.process_image(self.init_joint_angles)

	def process_text_prompts(self, ref_prompt, tar_prompt):
		text_tokens = self.processor([ref_prompt, tar_prompt], return_tensors="pt", padding=True).to(self.device)
		text_features = self.clip_model.get_text_features(**text_tokens)
		text_features = torch.nn.functional.normalize(text_features, dim=-1)
		self.text_features = text_features
		text_diff = text_features[1] - text_features[0]
		self.text_diff = torch.nn.functional.normalize(text_diff, dim=-1)
	
	def process_image(self, joints_angles):
		image_feature_list = []
		for view_idx in range(self.n_views):
			image = self.render(joints_angles, view_idx)
			image_feature = self.clip_model.get_image_features(
				# **self.processor(images=image, return_tensors="pt", padding=True, do_rescale=False).to(self.device)
				pixel_values=self.clip_transform(image).unsqueeze(0)
			).squeeze()
			image_feature = torch.nn.functional.normalize(image_feature, dim=-1)
			image_feature_list.append(image_feature)
		image_feature_list = torch.stack(image_feature_list)
		return image_feature_list
 
	def get_CLIP_dir_loss(self, joint_angle):
		image_feature = self.process_image(joint_angle)
		image_diff = image_feature - self.init_image_feature.detach()
		image_diff = torch.nn.functional.normalize(image_diff, dim=-1)
		return 1-(image_diff @ self.text_diff).mean()
	
	def get_CLIP_sim_loss(self, joint_angle):
		image_feature = self.process_image(joint_angle)
		cosine_sim = (image_feature @ self.text_features.T.detach()).softmax(dim=-1).mean(dim=0)[0]
		return 1-cosine_sim
	
	def target_func_bayes(self, x):
		joint_angles = self.init_joint_angles.clone()
		joint_angles[self.valid_screw_mask] = torch.tensor(x).to(self.device)
		return self.get_CLIP_loss(joint_angles).item()
	
	def bayes_optimize(self):
		self.valid_screw_mask = self.gaussians.get_screw_confs > 0.1
		search_range = []
		for i in range(self.valid_screw_mask.sum()):
			search_range.append((self.joint_limits[self.valid_screw_mask, 0][i].item(), self.joint_limits[self.valid_screw_mask, 1][i].item()))
		result = gp_minimize(self.target_func_bayes,       
				  search_range,      
				  acq_func="EI",      
				  n_calls=50,     
				  n_random_starts=10)
		
		out_joint_angles = self.init_joint_angles.clone()
		out_joint_angles[self.valid_screw_mask] = torch.tensor(result.x).to(self.device)
		
		return out_joint_angles
	
	def gradient_descent_optimize(self):
		
		lower = self.joint_limits[:, 0].detach()
		upper = self.joint_limits[:, 1].detach()
		# joint angles
		def x_to_angle(x):
			return torch.sigmoid(x) * (upper-lower) + lower
		
		x = torch.zeros_like(self.init_joint_angles).cuda().requires_grad_()
		optimizer = torch.optim.Adam([x], lr=0.01)

		for i in range(100):
			joint_angle = x_to_angle(x)
			loss = self.get_CLIP_loss(joint_angle)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % 10 == 0:
				print("iter : ", i, "loss : ", loss.item())
		
		return joint_angle
	
	def load_modules(self, clip_model="openai/clip-vit-large-patch14"):
		self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
		self.processor = CLIPProcessor.from_pretrained(clip_model)
		self.clip_transform = transforms.Compose([
			transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.CenterCrop(224),
			transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
		])

	def render(self, joint_angle, camera_idx):

		# render
		render_pkg = render_with_screw(
			self.cameras[camera_idx], self.gaussians, None, self.bg, 
			use_trained_exp=None, separate_sh=None, activate_screw_thres=None,
			desired_joint_angle=joint_angle,
			)
		image = torch.clamp(render_pkg["render"], 0.0, 1.0)

		return image # (3, 480, 640)

	def visualization(self, target_joint_angles, plot_idx=2, save=False):

		init_image = self.render(self.init_joint_angles, plot_idx)
		result_image = self.render(target_joint_angles, plot_idx)
		if save:
			imageio.imwrite('before.png', (init_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
			imageio.imwrite('after.png', (result_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
		plt.subplot(1, 2, 1)
		plt.imshow(init_image.permute(1, 2, 0).detach().cpu().numpy())
		plt.title("Before")
		plt.subplot(1, 2, 2)
		plt.imshow(result_image.permute(1, 2, 0).detach().cpu().numpy())
		plt.title("After")
		plt.show()

	def process_data(self, args, image_size=[640, 480]):

		# process args
		model_path = args.model_path
		white_background = args.white_background
		sh_degree = args.sh_degree

		# background color
		bg_color = [1, 1, 1] if white_background else [0, 0, 0]
		self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

		# get recent checkpoint
		checkpoint_name_list = [file for file in os.listdir(model_path) if file.endswith('.pth')]
		sorted_checkpoints_names = sorted(checkpoint_name_list, key=lambda x: int(x.replace('chkpnt', '').replace('.pth', '')))
		checkpoint_name = sorted_checkpoints_names[-1]

		# initialize gaussian        
		self.gaussians = ScrewGaussianModel(sh_degree)
		checkpoint = os.path.join(model_path, checkpoint_name)
		(model_params, first_iter) = torch.load(checkpoint, weights_only=False)
		self.gaussians.restore(model_params)

		# get joint limit
		joint_angles = self.gaussians.get_joint_angles
		lower_limit = torch.min(torch.stack(joint_angles), dim=0)[0]
		upper_limit = torch.max(torch.stack(joint_angles), dim=0)[0]
		self.joint_limits = torch.cat(
			(lower_limit.unsqueeze(-1), upper_limit.unsqueeze(-1)), dim=1)

		# load extrinsics and intrinsics
		intrinsic = np.load(os.path.join('control', 'intrinsic.npy'))
		fx = intrinsic[0, 0]
		fy = intrinsic[1, 1]
		cx = intrinsic[0, 2]
		cy = intrinsic[1, 2]

		# load extrinsics
		extrinsic_names = os.listdir(os.path.join('control', 'extrinsics'))
		extrinsic_names.sort()
		self.n_views = len(extrinsic_names)

		# process camera info
		self.cameras = []
		for idx, name in enumerate(extrinsic_names):

			# get camera to world frame
			extrinsic = np.load(os.path.join('control', 'extrinsics', name))

			# get the world-to-camera transform and set R, T
			R = np.transpose(extrinsic[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
			T = extrinsic[:3, 3]

			# fov
			FoVx = 2 * np.arctan(image_size[0] / (2 * fy))
			FoVy = 2 * np.arctan(image_size[1] / (2 * fx))

			# world view transform
			world_view_transform = torch.tensor(
				getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()

			self.cameras.append(
				MiniCam(
					FoVx=FoVx, FoVy=FoVy, image_width=image_size[0],
					image_height=image_size[1], cx=cx, cy=cy, art_idx=0,
					world_view_transform=world_view_transform
				)
			)

if __name__ == "__main__":

	# parser
	parser = ArgumentParser(description="Testing script parameters")
	parser.add_argument("--model_path", type=str)
	args = get_combined_args(parser)

	# prompt
	ref_prompt = "opened"
	tar_prompt = "closed"

	# model load
	model = Text2JointAngles(args, ref_prompt=ref_prompt, tar_prompt=tar_prompt)