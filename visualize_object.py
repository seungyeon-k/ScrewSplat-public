import numpy as np
import argparse
from omegaconf import OmegaConf
from articulated_object import ArticulatedObject
from articulated_object.get_camera_poses import get_camera_poses

if __name__ == '__main__':

	# load object infos
	object_infos = OmegaConf.load('configs/object_infos.yml')

	# argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/vis_obj_config.yml')
	args, unknown = parser.parse_known_args()
	cfg = OmegaConf.load(args.config)

	# object information
	object_class = cfg.object_class
	model_id = object_infos[object_class].model_id
	scale = object_infos[object_class].scale

	# articulation
	joint_angle = cfg.joint_angle

	# camera information
	camera_info = cfg.camera_info
	view_name = camera_info['view_name']
	radius = camera_info['radius']
	num_phi = camera_info['num_phi']
	phi_range = np.array(camera_info['phi_range']) / 180 * np.pi
	num_theta = camera_info['num_theta']
	theta_range = np.array(camera_info['theta_range']) / 180 * np.pi	

	# camera poses
	camera_poses = get_camera_poses(
		num_phi=num_phi,
		phi_range=phi_range,
		num_theta=num_theta,
		theta_range=theta_range,
		radius=radius
	)

	# load articulated object
	articulated_object = ArticulatedObject(
		model_id, scale=scale)

	# interactive visualizer
	articulated_object.visualize_object(
		camera_poses=camera_poses, theta=joint_angle)