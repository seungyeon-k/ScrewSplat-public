import torch
import numpy as np
import open3d as o3d
from copy import deepcopy
from robot.utils import *
from robot.openchains_torch import forward_kinematics

def compute_S_screw(A_screw, initialLinkFrames_from_base):
	"""_summary_
	Args:
			A_screw (torch.tensor): (n_joints, 6)
			initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
	"""
	S_screw = []
	for M, A in zip(initialLinkFrames_from_base, A_screw):
			S_temp = Adjoint(M.unsqueeze(0)).squeeze(0)@A.unsqueeze(-1)
			S_screw.append(S_temp)
	S_screw = torch.cat(S_screw, dim=-1).permute(1, 0)
	return S_screw # (n_joints, 6)

class Franka:

	def __init__(
			self, device='cpu', 
			load_links=False, load_d435=False, load_gripper=False):

		# colors
		self.robot_color = [0.9, 0.9, 0.9]
		self.azure_color = [0.4, 0.4, 0.4]
		self.bracket_color = [0.1, 0.1, 0.1]
		self.hand_color = [0.7, 0.7, 0.7]
		self.finger_color = [0.3, 0.3, 0.3]

		self.A_screw = torch.tensor([
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
		], dtype=torch.float32).to(device) 
		
		
		self.M = torch.zeros(7, 4, 4).to(device) 
		self.M[0] = torch.tensor([[1, 0, 0, 0], 
				[0, 1, 0, 0], 
				[0, 0, 1, 0.333], 
				[0, 0, 0, 1]]).to(device) 

		self.M[1] = torch.tensor([[1, 0, 0, 0], 
				[0, 0, 1, 0], 
				[0,-1, 0, 0], 
				[0, 0, 0, 1.0]]).to(device) 

		self.M[2] = torch.tensor([[1, 0, 0, 0], 
				[0, 0, -1, -0.316], 
				[0, 1, 0, 0], 
				[0, 0, 0, 1]]).to(device) 

		self.M[3] = torch.tensor([[1, 0, 0, 0.0825], 
				[0, 0,-1, 0], 
				[0, 1, 0, 0], 
				[0, 0, 0, 1]]).to(device) 

		self.M[4] = torch.tensor([[1, 0, 0, -0.0825], 
				[0, 0, 1, 0.384], 
				[0,-1, 0, 0], 
				[0, 0, 0, 1]]).to(device) 
			
		self.M[5] = torch.tensor([[1, 0, 0, 0], 
				[0, 0,-1, 0], 
				[0, 1, 0, 0], 
				[0, 0, 0, 1.0]]).to(device) 

		self.M[6] = torch.tensor([[1, 0, 0, 0.088], 
				[0, 0, -1, 0], 
				[0, 1, 0, 0], 
				[0, 0, 0, 1]]).to(device) 

		self.initialLinkFrames_from_base = torch.zeros(7, 4, 4).to(device) 
		self.initialLinkFrames_from_base[0] = self.M[0]
		for i in range(1, 7):
			self.initialLinkFrames_from_base[i] = self.initialLinkFrames_from_base[i-1]@self.M[i]

		# load link meshes
		if load_links:
			self.link_meshes = [None] * 8

			for link_num in range(8):
				self.link_meshes[link_num] = o3d.io.read_triangle_mesh(
					f"assets/panda/meshes/link{link_num}.ply")
				self.link_meshes[link_num].compute_vertex_normals()
				self.link_meshes[link_num].paint_uniform_color(self.robot_color)		

		# load camera meshes
		if load_d435:
			d = 0.008
			self.bracket_mesh = o3d.io.read_triangle_mesh("assets/panda/meshes/bracket.obj")
			self.bracket_mesh.compute_vertex_normals()
			self.bracket_mesh.paint_uniform_color(self.bracket_color)
			# self.bracket.translate([0, 0, - 0.002 - d])
			
			self.camera_mesh = o3d.io.read_triangle_mesh("assets/panda/meshes/d435.obj")
			self.camera_mesh.compute_vertex_normals()
			self.camera_mesh.paint_uniform_color(self.azure_color)
			roll = np.radians(0)
			pitch = np.radians(0)
			yaw = np.radians(-180)
			R = rpy_to_rotation_matrix(roll, pitch, yaw)
			self.camera_mesh = self.camera_mesh.rotate(R, center=np.array([0,0,0]))
		
		# load gripper
		if load_gripper:
			self.hand = o3d.io.read_triangle_mesh("assets/panda/meshes/hand.ply")
			self.hand.compute_vertex_normals()
			self.hand.paint_uniform_color(self.hand_color)
			self.finger1 = o3d.io.read_triangle_mesh("assets/panda/meshes/finger.ply")
			self.finger1.compute_vertex_normals()
			self.finger1.paint_uniform_color(self.finger_color)
			self.finger2 = o3d.io.read_triangle_mesh("assets/panda/meshes/finger.ply")
			self.finger2.compute_vertex_normals()
			self.finger2.paint_uniform_color(self.finger_color)
			
			# self.EEtoLeftFinger = torch.tensor(
			#         [[1, 0, 0, 0], 
			#         [0, 1, 0, 0], 
			#         [0, 0, 1, 0.0584], 
			#         [0, 0, 0, 1]]).to(device)
			
			# self.EEtoRightFinger = torch.tensor(
			#         [[-1, 0, 0, 0], 
			#         [0, -1, 0, 0], 
			#         [0, 0, 1, 0.0584], 
			#         [0, 0, 0, 1]]).to(device)
		
		self.LLtoEE = torch.tensor(
				[[0.7071, 0.7071, 0, 0], 
				[-0.7071, 0.7071, 0, 0], 
				[0, 0, 1, 0.107], 
				[0, 0, 0, 1]]).to(device)

		self.initialEEFrame = self.initialLinkFrames_from_base[-1]@self.LLtoEE
		
		self.S_screw = compute_S_screw(
				self.A_screw, 
				self.initialLinkFrames_from_base
		)
		
		self.inertias = torch.zeros(7, 6, 6).to(device)
				
		#############################################
		#################### Limits #################
		#############################################
		# https://frankaemika.github.io/docs/control_parameters.html
		# self.JointPos_Limits = [
		#         [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], 
		#         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.752, 2.8973]]
		
		offset = 0.02
		self.JointPos_Limits = [
		[-2.8973+offset, -1.7628+offset, -2.8973+offset, -3.0718+offset, -2.8973+offset, -0.0175+offset, -2.8973+offset], 
		[2.8973-offset, 1.7628-offset, 2.8973-offset, -0.0698-offset, 2.8973-offset, 3.752-offset, 2.8973-offset]]
		self.JointVel_Limits = [
				2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
		self.JointAcc_Limits = [
				15, 7.5, 10, 12.5, 15, 20, 20
		]
		self.JointJer_Limits = [
				7500, 3750, 5000, 6250, 7500, 10000, 10000
		]
		self.JointTor_Limits = [
				87, 87, 87, 87, 12, 12, 12
		]
		
		self.CarVelocity_Limits = [
				2*2.5, 2*1.7 
		]

	def get_robot_mesh(self, joint_angle, gripper_width=0.0):
		
		# forward kinematics
		_, LinkFrames_from_base, EEFrame = forward_kinematics(
			joint_angle.unsqueeze(0), 
			self.S_screw,
			self.initialLinkFrames_from_base,
			self.initialEEFrame)
		LinkFrames_from_base = LinkFrames_from_base.squeeze(0)
		EEFrame = EEFrame.squeeze(0)

		# load meshes
		robot_mesh = o3d.geometry.TriangleMesh()
		robot_mesh += self.link_meshes[0]
		for i in range(7):
			mesh = deepcopy(self.link_meshes[i+1])
			mesh.transform(LinkFrames_from_base[i])
			robot_mesh += mesh

		# end-effector to camera
		ee2bracket = torch.eye(4)
		ee2bracket[:3, 3] = torch.tensor([0, 0, -0.002])
		ee2cam = torch.eye(4)
		ee2cam[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
		ee2cam[:3, 3] = torch.tensor([0.069, 0, 0.008])

		# camera
		mesh = deepcopy(self.bracket_mesh)
		mesh.transform(EEFrame @ ee2bracket)
		robot_mesh += mesh
		mesh = deepcopy(self.camera_mesh)
		mesh.transform(EEFrame @ ee2cam)
		robot_mesh += mesh

		# end-effector to fingers
		bracket_offset = torch.eye(4)
		bracket_offset[2, 3] = 0.008
		ee2finger1 = torch.eye(4)
		ee2finger1[:3, 3] = torch.tensor([0, gripper_width / 2, 0.1654 / 3])
		ee2finger2 = torch.eye(4)
		ee2finger2[:3, :3] = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
		ee2finger2[:3, 3] = torch.tensor([0, - gripper_width/2, 0.1654 / 3])		

		# gripper
		mesh = deepcopy(self.hand)
		mesh.transform(EEFrame @ bracket_offset)
		robot_mesh += mesh
		mesh = deepcopy(self.finger1)
		mesh.transform(EEFrame @ bracket_offset @ ee2finger1)
		robot_mesh += mesh
		mesh = deepcopy(self.finger2)
		mesh.transform(EEFrame @ bracket_offset @ ee2finger2)
		robot_mesh += mesh

		return robot_mesh


def rpy_to_rotation_matrix(roll, pitch, yaw):
		# Calculate the cosine and sine of each angle
	cr = np.cos(roll)
	sr = np.sin(roll)
	cp = np.cos(pitch)
	sp = np.sin(pitch)
	cy = np.cos(yaw)
	sy = np.sin(yaw)

	# Define the rotation matrix components
	R = np.array([
		[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
		[sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
		[-sp, cp * sr, cp * cr]
	])
	
	return R