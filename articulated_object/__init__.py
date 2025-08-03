import numpy as np
import torch
import os
import trimesh
from lxml import etree as ET
from copy import deepcopy
from articulated_object.utils import (
	Adjoint, exp_se3, parse_vector_to_tensor, xyz_rpy_to_SE3
)

class ArticulatedObject:
	def __init__(self, model_id, scale=1.0):
		
		# arguments
		self.scale = scale 

		# load urdf file
		self.dir_load = os.path.join('datasets', 'partnet_mobility', str(model_id))
		urdf_file = os.path.join(self.dir_load, 'mobility.urdf')
		tree = ET.parse(urdf_file)
		self.root = tree.getroot()

		# process object
		self.get_object()

	#############################################################
	######################## VISUALIZER #########################
	#############################################################		

	def visualize_object(self, camera_poses=None, theta=0.0):

		# visualizer
		scene = trimesh.Scene()

		# mesh list
		meshes = []

		# visualize object
		thetas = torch.ones(len(self.S_screws.keys())) * theta
		mesh = self.update_object(thetas)
		global_frame = trimesh.creation.axis(
			origin_size=0.02, transform=np.eye(4))
		meshes.append(mesh)
		meshes.append(global_frame)

		# visualize camera SE3s
		if camera_poses is not None:
			for pose in camera_poses:
				frame = trimesh.creation.axis(origin_size=0.02, transform=pose)
				meshes.append(frame)		

		# render
		for mesh in meshes:
			scene.add_geometry(mesh)
		scene.show(background=np.array([1.0, 1.0, 1.0]))

	#############################################################
	######################## UPDATE OBJECT ######################
	#############################################################

	def update_object(self, theta, return_link_type=False):

		# update poses
		self.forward_kinematics(theta)

		# find base
		if return_link_type:
			for base_link_name, link in self.links.items():
				if link['parent'] == None:
					break
			for static_link_name, link in self.links.items():
				if link['parent'] == base_link_name:
					break

		# get meshes
		meshes = []
		if return_link_type:
			link_types = []
		for i, (link_name, link) in enumerate(self.links.items()):
			link_mesh = deepcopy(link['mesh'])
			if link_mesh is None:
				continue
			for mesh in link_mesh:
				mesh.apply_transform(self.poses[link_name].numpy())
			meshes.append(link_mesh)
			if return_link_type:
				if link_name == static_link_name:
					link_types.append('static')
				else:
					link_types.append('movable')

		# return
		if return_link_type:
			return meshes, link_types
		else:
			return meshes

	def forward_kinematics(self, theta):

		# initialize
		if len(theta) != len(self.S_screws.keys()):
			raise ValueError('The length of theta does not match with the length of screws.')
		screw_exponentials = dict()
		self.poses = deepcopy(self.zero_poses)

		# theta dictionary
		keys = self.S_screws.keys()
		theta_dict = {key: value for key, value in zip(keys, theta)}

		# find base
		for base_link_name, link in self.links.items():
			if link['parent'] == None:
				break

		# screw exponential for base
		screw_exponentials[base_link_name] = torch.eye(4)

		# update function
		def update_children_poses(parent):

			# children
			children = self.links[parent]['children']
			
			# children
			for child in children:

				# S screw update
				exist_joint = False
				for screw_name, screw_info in self.S_screws.items():
					if screw_info['child'] == child:
						screw_theta = screw_info['screw'] * theta_dict[screw_name]
						screw_theta_exp = exp_se3(screw_theta)
						screw_exponentials[child] = screw_exponentials[parent] @ screw_theta_exp
						exist_joint = True
				if not exist_joint:
					screw_exponentials[child] = screw_exponentials[parent]

				# pose update
				self.poses[child] = screw_exponentials[child] @ self.zero_poses[child]

				# update child's children zero poses
				update_children_poses(child)

		# update poses
		update_children_poses(base_link_name)

	#############################################################
	####################### LOAD OBJECT #########################
	#############################################################

	def get_object(self):

		self.get_joints()
		self.get_links()
		self.get_screws()
		self.get_zero_poses()

	def get_zero_poses(self):
		
		# initialize
		self.zero_poses = dict()
		self.S_screws = deepcopy(self.A_screws)
		for link_name, link in self.links.items():
			self.zero_poses[link_name] = link['SE3']

		# find base
		for base_link_name, link in self.links.items():
			if link['parent'] == None:
				break

		# update function
		def update_children_poses(parent):
			
			# children
			children = self.links[parent]['children']
			
			# children
			for child in children:
				
				# zero pose update
				self.zero_poses[child] = self.zero_poses[parent] @ self.zero_poses[child]

				# S screw update
				for _, screw_info in self.S_screws.items():
					if screw_info['parent'] == child:
						screw_info['screw'] = Adjoint(self.zero_poses[child]) @ screw_info['screw']

				# update child's children zero poses
				update_children_poses(child)

		# update poses
		update_children_poses(base_link_name)

	def get_screws(self):
		
		# initialize
		self.A_screws = dict()

		# update screws
		for joint_name, joint in self.joints.items():

			# revolute or prismatic
			joint_type = joint['type']
			if joint_type not in ['revolute', 'prismatic', 'continuous', 'fixed']:
				raise NotImplementedError
			if joint_type == 'fixed':
				continue

			# initialize
			screw_info = dict()

			# A vector
			axis = joint['axis']
			xyz = joint['origin']['xyz']
			if joint_type == 'revolute' or joint_type == 'continuous':
				screw_info['screw'] = torch.cat([axis, -torch.cross(axis, xyz, dim=0)])
			elif joint_type == 'prismatic':
				screw_info['screw'] = torch.cat([torch.zeros(3), axis])

			# info
			screw_info['parent'] = joint['parent']
			screw_info['child'] = joint['child']
			if joint_type == 'continuous':
				screw_info['limit'] = torch.tensor([-3.14, 3.14])
			elif joint_type == 'prismatic':
				screw_info['limit'] = torch.tensor([
					joint['limit']['lower'] * self.scale, 
					joint['limit']['upper'] * self.scale]) 
			elif joint_type == 'revolute':
				screw_info['limit'] = torch.tensor([
					joint['limit']['lower'], 
					joint['limit']['upper']]) 
			# update screws
			self.A_screws[joint_name] = screw_info

	def get_links(self):

		# initialize
		self.links = dict()

		# iterate link
		for link in self.root.iter('link'):
			
			# link name
			link_name = link.get('name')

			# link pose
			SE3_link = torch.eye(4)

			# link initialize
			link_mesh = []

			# link mesh
			if link.find('visual') is not None:
				for i, visual in enumerate(link.iter('visual')):

					# origin
					origin = visual.find('origin')
					xyz = parse_vector_to_tensor(origin.get('xyz', '0 0 0'))
					rpy = parse_vector_to_tensor(origin.get("rpy", "0 0 0"))
					SE3 = xyz_rpy_to_SE3(xyz, rpy)

					# mesh
					geometry = visual.find('geometry')
					mesh = geometry.find('mesh')
					filename = mesh.get('filename')
					part_mesh = trimesh.load_mesh(os.path.join(self.dir_load, filename))
					
					# mesh transform and scaling
					scale_matrix = np.eye(4)
					scale_matrix[:3, :3] = np.diag([self.scale, self.scale, self.scale])
					part_mesh.apply_transform(SE3.numpy())
					part_mesh.apply_transform(scale_matrix)
					
					link_mesh.append(part_mesh)
				
			else:
				link_mesh = None
				SE3_link = torch.eye(4)

			# tree
			parent = None
			children = []
			for _, joint_info in self.joints.items():
				if joint_info['parent'] == link_name:
					children.append(joint_info['child'])
				if joint_info['child'] == link_name:
					parent = joint_info['parent']
					SE3_joint = xyz_rpy_to_SE3(
						joint_info['origin']['xyz'], joint_info['origin']['rpy'])
					SE3_link = SE3_joint @ SE3_link 

			# link info
			self.links[link_name] = {
				'mesh': link_mesh,
				'SE3': SE3_link,
				'children': children,
				'parent': parent
			}

	def get_joints(self):
		
		# initialize
		self.joints = dict()

		# iterate joint
		for joint in self.root.iter('joint'):

			# joint info
			joint_info = {}
			joint_name = joint.get('name')
			joint_info['type'] = joint.get('type')

			# extract origin info
			origin = joint.find("origin")
			if origin is not None:
				joint_info['origin'] = {
					'xyz': parse_vector_to_tensor(origin.get("xyz", "0 0 0")) * self.scale,
					'rpy': parse_vector_to_tensor(origin.get("rpy", "0 0 0"))
				}
			else:
				joint_info['origin'] = None

			# extract axis info
			axis = joint.find("axis")
			joint_info['axis'] = parse_vector_to_tensor(axis.get("xyz")) if axis is not None else None

			# extract parent and child
			parent = joint.find("parent")
			joint_info['parent'] = parent.get('link') if parent is not None else None
			child = joint.find("child")
			joint_info['child'] = child.get('link') if child is not None else None

			# extract limit
			limit = joint.find("limit")
			if limit is not None:
				joint_info['limit'] = {
					'lower': float(limit.get("lower")),
					'upper': float(limit.get("upper")),
				}
			else:
				joint_info['limit'] = None

			self.joints[joint_name] = joint_info

if __name__ == '__main__':

	# object
	model_id = '10211'

	# main
	articulated_object = ArticulatedObject(model_id)

	
