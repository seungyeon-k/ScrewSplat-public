import torch
from tqdm import tqdm
from robot.utils import *

######################################
# A_screw: (n_joints, 6)             #  
# init link frames: (n_joints, 4, 4) #
# init eeframe: (4, 4)               #
# inertias: (n_joints, 6, 6)         #
######################################

def inverse_kinematics(
		initjointPos, 
		desiredEEFrame,
		robot,
		max_iter=5000, 
		step_size1=0.01, 
		step_size2=0.0001, 
		tolerance=0.001,
		locked_joints=[None]*7,
		device='cpu',
		enable_pbar=True
	):
	"""_summary_

	Args:
		initjointPos (torch.tensor): (bs, dof)
		desiredEEFrame (torch.tensor): (bs, 4, 4)
		robot (robot class): robot class
		max_iter (int, optional): Defaults to 5000.
		step_size1 (float, optional): Defaults to 0.01.
		step_size2 (float, optional): Joint limit. Defaults to 0.0001.
		step_size3 (float, optional): Self-collision. Defaults to 0.0.
		tolerance (float, optional): Defaults to 0.001.
	"""

	# initialize
	bs, dof = initjointPos.size()
	jointPos = initjointPos.clone()
	iter_ = 0
	jl_eps = 0.01
	max_error = torch.inf
	min_JL_bool = False
	COL_bool = True

	# joint locking
	if len(locked_joints) != dof:
		raise ValueError('shape of locked joints should be match to joints')
	locked_joint_idxs = [i for i in range(dof) if locked_joints[i] is not None]
	unlocked_joint_idxs = [i for i in range(dof) if locked_joints[i] is None]
	locked_joints = torch.tensor(
		[0.0 if locked_joints[i] is None else locked_joints[i] for i in range(dof)]
	)
	n_unlocked = len(unlocked_joint_idxs)

	# joint limit
	joint_limit_thr = torch.tensor(robot.JointPos_Limits).to(device) # (2, 7)
	joint_limit_thr = joint_limit_thr[:, unlocked_joint_idxs]
	joint_limit_thr[0, :] += jl_eps 
	joint_limit_thr[1, :] -= jl_eps

	# robot variables
	S_screw = robot.S_screw
	initialLinkFrames_from_base = robot.initialLinkFrames_from_base
	initialEEFrame = robot.initialEEFrame

	# pbar
	if enable_pbar:
		pbar = tqdm(
			total=max_iter, 
			desc=f"solving inverse kinematics... ", 
			leave=False
		)	

	while (
			(iter_ < max_iter) and 
			((max_error > tolerance) or (not min_JL_bool) or (not COL_bool))
		):
		
		# initialize
		jointPos[:, locked_joint_idxs] = locked_joints[locked_joint_idxs].unsqueeze(0).repeat(bs, 1).to(jointPos)

		# forward kinematics
		expSthetas, _, EEFrame = forward_kinematics(
			jointPos, 
			S_screw,
			initialLinkFrames_from_base,
			initialEEFrame)

		# get body jacobian
		BodyJacobian = get_BodyJacobian(
			S_screw, expSthetas, EEFrame)
		BodyJacobian = BodyJacobian[:, :, unlocked_joint_idxs]
		
		# error vector
		dR = EEFrame[:, :3, :3].permute(0, 2, 1) @ desiredEEFrame[:, :3, :3]
		wb = skew(log_SO3(dR)).unsqueeze(-1) # (bs, 3, 1)
		vb = (EEFrame[:, :3, :3].permute(0, 2, 1) @ (
			desiredEEFrame[:, :3, 3:] - EEFrame[:, :3, 3:])) # (bs, 3, 1)
		error_vector = torch.cat([wb, vb], dim=1)
		
		# pullback error to joint
		inv_Jb = approxmiate_pinv(BodyJacobian)
		jointVel = (inv_Jb @ error_vector).squeeze(-1)
		grad1 = step_size1 * jointVel
	
		if step_size2 != 0:
			
			# null space projection
			eye_unlocked = torch.eye(n_unlocked, device=jointPos.device)
			null_proj = eye_unlocked.unsqueeze(0) - inv_Jb @ BodyJacobian

			# check joint limit
			low_violation_idx = joint_limit_thr[0:1, :] > jointPos[:, unlocked_joint_idxs]
			high_violation_idx = joint_limit_thr[1:2, :] < jointPos[:, unlocked_joint_idxs]
			
			# joint limit preventing vector
			Sigma = torch.zeros(bs, n_unlocked).to(jointPos)
			Sigma[low_violation_idx] = + (
				jointPos[:, unlocked_joint_idxs][low_violation_idx]
				- joint_limit_thr[0:1].repeat(bs, 1)[low_violation_idx]) ** 2 / jl_eps ** 2
			Sigma[high_violation_idx] = - (
				jointPos[:, unlocked_joint_idxs][high_violation_idx]
				- joint_limit_thr[1:2].repeat(bs, 1)[high_violation_idx]) ** 2 / jl_eps ** 2

			# get gradient
			grad2 = step_size2 * null_proj @ Sigma.unsqueeze(-1)
			grad2 = grad2.squeeze(-1)
		
		# update
		if step_size2 != 0:
			jointPos[:, unlocked_joint_idxs] += (grad1 + grad2)
		else:
			jointPos[:, unlocked_joint_idxs] += grad1

		# iteration
		iter_ += 1

		# error check
		error = ((EEFrame - desiredEEFrame) ** 2).sum(-1).sum(-1)
		min_error = error[~error.isnan()].min()
		max_error = error[~error.isnan()].max()
		
		# joint limit check
		JointPos_Limits = torch.tensor(robot.JointPos_Limits).to(device)
		JL_bool = (
			(jointPos - JointPos_Limits[0:1] > 0) * (
				JointPos_Limits[1:2] - jointPos > 0)
		).prod(dim=-1)
		min_JL_bool = bool(JL_bool.prod())
		
		# pbar update
		if enable_pbar:
			if iter_ % 100 == 0:
				pbar.update(100)
				print_str = {
					'min error': min_error.item(),
					'max error': max_error.item(),
					'min joint limit': min_JL_bool
				}
				pbar.set_postfix(**print_str)

	# close tqdm
	if enable_pbar:
		pbar.close()
	   
	# change by SY
	dict_infos = {
		'final_error': error,
		'joint limit check': JL_bool.bool(),
		"self-collision check": {COL_bool}
	}

	return jointPos, dict_infos

def forward_kinematics(
		jointPos, 
		S_screw, 
		initialLinkFrames_from_base, 
		initialEEFrame, 
		link_frames=True):
	"""_summary_
	Args:
		jointPos (torch.tensor): (bs, n_joints)
		S_screw (torch.tensor): (n_joints, 6)
		initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
		M_01, M_02, ..., M_0n
		initialEEFrame (torch.tensor): (4, 4)
		M_0b
	"""

	# initialize
	n_joints = len(S_screw)
	bs = len(jointPos)

	# screw exponential
	screw_theta = (jointPos.unsqueeze(-1) * S_screw.unsqueeze(0)).reshape(-1, 6)
	screw_exponentials = exp_se3(screw_theta).reshape(bs, n_joints, 4, 4)
	
	# exponential matrices
	expSthetas = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(bs, n_joints+1, 1, 1).to(jointPos)
	for i in range(n_joints):
		expSthetas[:, i+1, :, :] = (
			expSthetas[:, i, :, :] @ screw_exponentials[:, i, :, :])
	expSthetas = expSthetas[:, 1:, :, :]

	# end-effector frame
	EEFrame = expSthetas[:, -1] @ initialEEFrame.unsqueeze(0) # (bs, 4, 4)
	if link_frames:
		LinkFrames_from_base = expSthetas @ initialLinkFrames_from_base.unsqueeze(0)
		return expSthetas, LinkFrames_from_base, EEFrame # (bs, n_joints, 4, 4), (bs, 4, 4)
	else: 
		return EEFrame # (bs, 4, 4)

def get_SpaceJacobian(S_screw, expSthetas):
	"""_summary_
	Args:
		S_screw (torch.tensor): (n_joints, 6)
		expSthetas (torch.tensor): (bs, n_joints, 4, 4)
	"""

	bs, n_joints, _, _ = expSthetas.size()

	# adjoint mapping
	adjoint_mappings = Adjoint(
		expSthetas.reshape(-1, 4, 4)
	).reshape(bs, n_joints, 6, 6)
	adjoint_mappings = torch.cat(
		[
			torch.eye(6).unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1).to(expSthetas),
			adjoint_mappings[:, :-1, :, :]
		], dim=1) # bs, n_joints, 6, 6
	SpaceJacobian = adjoint_mappings @ S_screw.unsqueeze(0).unsqueeze(-1) # bs, n_joints, 6, 1
	SpaceJacobian = SpaceJacobian.squeeze(-1).permute(0, 2, 1) # bs, 6, n_joints

	return SpaceJacobian

def get_BodyJacobian(S_screw, expSthetas, EEFrame):
	"""_summary_
	Args:
		S_screw (torch.tensor): (n_joints, 6)
		expSthetas (torch.tensor): (bs, n_joints, 4, 4)
		EEFrame (torch.tensor): (bs, 4, 4)
	"""
	SpaceJacobian = get_SpaceJacobian(S_screw, expSthetas)
	BodyJacobian = Adjoint(inv_SE3(EEFrame)) @ SpaceJacobian
	return BodyJacobian # (bs, 6, n_joints)