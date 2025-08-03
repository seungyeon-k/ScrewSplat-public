import numpy as np
import argparse
from omegaconf import OmegaConf
from articulated_object.articulated_object_renderer import ArticulatedObjectRenderer
from articulated_object.get_camera_poses import get_camera_poses

if __name__ == '__main__':

	# load object infos
	object_infos = OmegaConf.load('configs/object_infos.yml')

	# argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/gen_data_config.yml')
	args, unknown = parser.parse_known_args()
	cfg = OmegaConf.load(args.config)

	# object information
	object_classes = cfg.object_classes

	# camera poses
	camera_info = cfg.camera_info
	view_name = camera_info['view_name']
	radius = camera_info['radius']
	num_phi = camera_info['num_phi']
	phi_range = np.array(camera_info['phi_range']) / 180 * np.pi
	num_theta = camera_info['num_theta']
	theta_range = np.array(camera_info['theta_range']) / 180 * np.pi	

	# camera intrinsics
	camera_intr = cfg.camera_intr

	# camera poses
	camera_poses = get_camera_poses(
		num_phi=num_phi,
		phi_range=phi_range,
		num_theta=num_theta,
		theta_range=theta_range,
		radius=radius
	)

	# rendering information
	render_info = cfg.render_info
	blender_root = render_info['blender_root']

	# dataset information
	dataset_info = cfg.dataset_info
	offset = dataset_info['offset']
	steps = dataset_info['steps']
	mode = dataset_info['mode']
	split_list = dataset_info['split_list']

	# data generation
	for object_class in object_classes:

		# object info
		category = object_class.split('-')[0]
		model_id = object_infos[object_class].model_id
		scale = object_infos[object_class].scale
		joint_indices = object_infos[object_class].get('joint_indices', None)
		joint_limits = object_infos[object_class].get('joint_limits', None)
		shadow_on = object_infos[object_class].get('shadow_on', False)

		# load articulated object
		articulated_object = ArticulatedObjectRenderer(
			model_id, category, blender_root, camera_intr, scale=scale)

		# render rgb image
		articulated_object.generate_mobility_dataset(
			camera_poses, 
			steps=steps, offset=offset,
			joint_indices=joint_indices,
			joint_custom_limits=joint_limits,
			shadow_on=shadow_on,
			mode=mode, split_list=split_list, view_name=view_name)