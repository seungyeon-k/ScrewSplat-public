"""
PyTorch-based SO(3) and SE(3) operations libs
"""
import numpy as np
import torch

def parse_vector_to_tensor(vector_str):

	if vector_str is None:
		return None

	return torch.tensor(np.fromstring(vector_str, dtype='float32', sep=' '))

def rpy_to_rotation_matrix(rpy):
	
	# roll pitch yaw
	roll, pitch, yaw = rpy
	
	# Rotation about x-axis (roll)
	Rx = torch.tensor([
		[1, 0, 0],
		[0, torch.cos(roll), -torch.sin(roll)],
		[0, torch.sin(roll), torch.cos(roll)]
	], dtype=torch.float32)

	# Rotation about y-axis (pitch)
	Ry = torch.tensor([
		[torch.cos(pitch), 0, torch.sin(pitch)],
		[0, 1, 0],
		[-torch.sin(pitch), 0, torch.cos(pitch)]
	], dtype=torch.float32)

	# Rotation about z-axis (yaw)
	Rz = torch.tensor([
		[torch.cos(yaw), -torch.sin(yaw), 0],
		[torch.sin(yaw), torch.cos(yaw), 0],
		[0, 0, 1]
	], dtype=torch.float32)

	# Combined rotation
	return Rz @ Ry @ Rx

def xyz_rpy_to_SE3(xyz, rpy):

	rotation_matrix = rpy_to_rotation_matrix(rpy)
	SE3 = torch.eye(4, dtype=torch.float32)
	SE3[:3, :3] = rotation_matrix
	SE3[:3, 3] = xyz
	return SE3

def b_mul(x, y):
	leny = len(y.size())
	lenx = len(x.size())
	if lenx > leny:
		for _ in range(lenx - leny):
			y = y.unsqueeze(-1)
		return x*y 
	else:
		for _ in range(leny - lenx):
			x = x.unsqueeze(-1)
		return x*y 
	
def b_div(x, y):
	leny = len(y.size())
	lenx = len(x.size())
	assert lenx >= leny
	for _ in range(lenx - leny):
		y = y.unsqueeze(-1)
	return x/y 
	
def skew(w):
	"""_summary_
	w <-> [w] / R^3 <-> so(3)
	v <-> [v] / R^6 <-> se(3)
	"""
	bs = w.size(0)
	if w.size() == (bs, 3, 3):
		W = torch.cat([
			-w[:, 1, 2:3],
			w[:, 0, 2:3],
			-w[:, 0, 1:2]
			], dim=1)
	elif w.size() == (bs, 3):
		zero1 = torch.zeros(bs, 1, 1).to(w)
		w = w.unsqueeze(-1).unsqueeze(-1)
		W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
					   torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
					   torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
	elif w.size() == (bs, 4, 4):
		w_so3 = w[:, :3, :3]
		W = torch.cat([skew(w_so3), w[:, :3, 3]], dim=1)
	elif w.size() == (bs, 6):
		w_so3 = w[:, :3]
		w_temp = torch.cat(
			[skew(w_so3), w[:, 3:].unsqueeze(-1)], dim=-1)
		zero1 = torch.zeros(bs, 1, 4).to(w)
		W = torch.cat([w_temp, zero1], dim=1)
	else:
		raise NotImplementedError
	return W

def exp_so3(x):
	"""_summary_
	x -> exp([x])
	or
	[x] -> exp([x])
	"""
	bs = x.size(0)
	if x.size() == (bs, 3, 3):
		W = x
		w = skew(W)
	elif x.size() == (bs, 3):
		w = x
		W = skew(w)
	else:
		raise NotImplementedError
	
	wnorm = torch.norm(w, dim=1)
	sw = torch.sin(wnorm)
	cw = torch.cos(wnorm)
	
	Id = torch.eye(3).unsqueeze(0).to(x)
	
	eps = 1e-7
	idx = wnorm > eps 
	
	R = torch.zeros(bs, 3, 3).to(x)
	R[~idx] = Id + W[~idx] + 0.5*(W[~idx]@W[~idx])
	W_hat = b_div(W[idx], wnorm[idx])
	R[idx] = Id + b_mul(sw[idx], W_hat) + b_mul((1-cw[idx]), W_hat@W_hat)
	return R

def proj_minus_one_plus_one(x):
	eps = 1e-6
	x = torch.min(x, (1 - eps) * (torch.ones(x.shape).to(x)))
	x = torch.max(x, (-1 + eps) * (torch.ones(x.shape).to(x)))
	return x

def log_SO3(R):
	"""_summary_
	R -> log(R)
	"""
	eps = 1e-4
	trace = torch.sum(R[:, range(3), range(3)], dim=1)

	omega = R * torch.zeros(R.shape).to(R)

	theta = torch.acos(proj_minus_one_plus_one((trace - 1) / 2))

	temp = theta.unsqueeze(-1).unsqueeze(-1)

	omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
		(torch.abs(trace + 1) > eps) * (theta > eps)]

	omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

	omega_vector_temp = torch.sqrt(omega_temp[:, range(3), range(3)] + torch.ones(3).to(R))
	A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
	B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
	C = omega_vector_temp[:, 0]
	omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
	omega[torch.abs(trace + 1) <= eps] = skew(omega_vector) * torch.pi
	return omega

def exp_se3(x):
	"""_summary_
	x = (w, v)
	||w|| \neq 0 
	x -> exp([x])
	or
	[x] -> exp([x])
	"""

	wo_batch = False
	if x.ndim == 1:
		wo_batch = True
		x = x.unsqueeze(0)

	bs = x.size(0)
	if x.size() == (bs, 4, 4):
		W = x[:, :3, :3]
		w = skew(W)
		v = x[:, :3, 3]
	elif x.size() == (bs, 6):
		w = x[:, :3]
		v = x[:, 3:]
		W = skew(w)
	else:
		raise NotImplementedError
	
	eps = 1e-8
	wnorm = torch.norm(w, dim=1)
	idx = wnorm < eps

	wnorm = wnorm.unsqueeze(-1).unsqueeze(-1)
	wnorm_clip = torch.clip(wnorm, min=eps)
	sw = torch.sin(wnorm)
	cw = torch.cos(wnorm)
	
	Id = torch.eye(3).unsqueeze(0).to(x)
	W_squared = W @ W
	W_idx = W[idx]
	W_squared_idx = W_squared[idx]
	
	# compute R
	R = Id + sw * W / wnorm_clip + (1 - cw) * W_squared / (wnorm_clip**2)
	R[idx] = Id + W_idx + 0.5 * W_squared_idx
	
	# compute G(wnorm)
	G = Id  + (1 - cw) * W / (wnorm_clip**2) + (wnorm - sw) * W_squared / (wnorm_clip**3)
	G[idx] = Id + 0.5 * W_idx + 1/6 * W_squared_idx
	p = G @ v.unsqueeze(-1)
	
	# construct T
	T = torch.cat([
		torch.cat([R, p], dim=-1),
		torch.tensor([[[0., 0., 0., 1.]]]).repeat(bs, 1, 1).to(x)
	], dim=1)

	if wo_batch:
		T = T.squeeze(0)

	return T


def log_SE3(T):
	"""_summary_
	T -> (w, v)
	"""
	bs = T.size(0)
	R = T[:, 0:3, 0:3]
	p = T[:, 0:3, 3:]  
	
	W = log_SO3(R)  
	w = skew(W)  

	wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(bs), range(bs)]]  
	wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  
	wnorm = torch.sqrt(wsqr)  
	wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  
	wnorm_inv = 1 / wnorm_unsqueezed  
	cw = torch.cos(wnorm).unsqueeze(-1).unsqueeze(-1)  
	sw = torch.sin(wnorm).unsqueeze(-1).unsqueeze(-1)  

	P = torch.eye(3).to(W) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (
				wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
	v = torch.inverse(P) @ p  # n,3,1
	return torch.cat([torch.cat([W, v], dim=2), torch.zeros(bs, 1, 4).to(W)], dim=1)

def inv_SE3(T):
	"""_summary_
	T -> T^-1
	"""
	bs = len(T)
	R = T[:, :3, :3]
	p = T[:, :3, 3:]
	
	temp = torch.cat([R.permute(0, 2, 1), -R.permute(0, 2, 1)@p], dim=-1)
	invT = torch.cat([temp, torch.tensor(bs*[[[0., 0., 0., 1.]]]).to(T)], dim=1)
	return invT

def Adjoint(T):
	"""_summary_
	T -> [Ad_T]
	"""

	wo_batch = False
	if T.ndim == 2:
		wo_batch = True
		T = T.unsqueeze(0)

	bs = len(T)
	R = T[:, :3, :3]
	p = T[:, :3, 3]
	zeros = torch.zeros(bs, 3, 3).to(T)
	AdT = torch.cat([
		torch.cat([R, zeros], dim=-1),
		torch.cat([skew(p) @ R, R], dim=-1)    
	], dim=1)

	if wo_batch:  
		AdT = AdT.squeeze(0)

	return AdT

def adjoint(V):
	"""_summary_
	V -> [ad_V]
	"""
	bs = len(V)
	w = V[:, :3]
	v = V[:, 3:]
	zeros = torch.zeros(bs, 3, 3).to(V)
	adV = torch.cat([
		torch.cat([skew(w), zeros], dim=-1),
		torch.cat([skew(v), skew(w)], dim=-1)    
	], dim=1)
	return adV

def approxmiate_pinv(J):
	"""_summary_

	Args:
		J (torch.tensor): (bs, 6, dof)
	"""
	bs, _, dof = J.size()
	temp = J@J.permute(0, 2, 1) + 1.0e-7*torch.eye(6).unsqueeze(0).to(J)
	return J.permute(0, 2, 1)@torch.inverse(temp)