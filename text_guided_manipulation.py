import torch
import os
from argparse import ArgumentParser
from arguments import get_combined_args
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from control.text_to_joints import Text2JointAngles
from control.trajectory_planner import TrajectoryPlanner

if __name__ == "__main__":

	# load object infos
	pretrained_infos = OmegaConf.load('configs/pretrained_infos.yml')

	# argparse
	parser = ArgumentParser(description="Testing script parameters")
	parser.add_argument("--object_class", type=str, default=None)
	args, unknown = parser.parse_known_args()
	if args.object_class:
		object_class = args.object_class
		model_path = pretrained_infos[object_class].model_path
		parser.add_argument("--model_path", type=str, default=model_path)
	else:
		parser.add_argument("--model_path", type=str)
	args = get_combined_args(parser)

	# prompts for foldingchair
	ref_prompt = "usable"
	tar_prompt = "folded"
	
	# prompts for laptop, usb
	# ref_prompt = "usable"
	# tar_prompt = "folded"
	
	# prompts for refrigerator
	# ref_prompt = "opened"
	# tar_prompt = "closed"

	# prompts for refrigerator-2
	# ref_prompt = "open drawer"
	# tar_prompt = "closed drawer"

	# run
	print("Initializing text2joint module.")
	text_to_joints = Text2JointAngles(
		args, ref_prompt=ref_prompt, tar_prompt=tar_prompt,
		joint_angle_idx=0) # joint_angle_idx can be 0, 1, 2, 3, 4 since 
						   # ScrewSplat is optimized with five configurations.
	
	# optimization
	text_to_joints.get_CLIP_loss = text_to_joints.get_CLIP_dir_loss
	out_joint_angles = text_to_joints.bayes_optimize()
	print(f"current joint angle: {text_to_joints.init_joint_angles}")
	print(f"optimized joint angle: {out_joint_angles}")
	
	# visualization
	text_to_joints.visualization(out_joint_angles, plot_idx=31, save=True)

	# trajectory planner
	print("Initializing robot trajectory planner module.")
	current_angle = text_to_joints.init_joint_angles.cpu()
	trajectory_planner = TrajectoryPlanner(
		args, center=[0.65, 0.0, 0.243], joint_angle=current_angle)

	# trajectory planning
	print("Planning robot trajectory.")
	target_angle = out_joint_angles.cpu()
	trajectory, tip_trajectory = trajectory_planner.trajectory_planning(
		target_angle, num_points=10, return_tip_trajectory=True,
		offset_ee_angle=-torch.pi/4)

	# visualization
	trajectory_planner.visualization(
		trajectory=trajectory, tip_trajectory=tip_trajectory)