import torch
import numpy as np
import os
import json
from torch import nn
from scipy.spatial import KDTree
from random import randint
from plyfile import PlyData, PlyElement

from articulated_object.utils import exp_se3
from articulated_object.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

#############################################################
#################### SCREW GAUSSIAN MODEL ###################
#############################################################
class ScrewGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def screw_from_vector(vector):
            w_revol = vector[:self.n_revol, :3]
            w_revol = w_revol / torch.norm(w_revol, dim=1, keepdim=True)
            v_revol = - torch.cross(w_revol, vector[:self.n_revol, 3:], dim=1) 
            v_pris = vector[self.n_revol:, :3]
            v_pris = v_pris / torch.norm(v_pris, dim=1, keepdim=True)
            return torch.cat(
                (
                    torch.cat((w_revol, v_revol), dim=1), 
                    torch.cat((torch.zeros(self.n_pris, 3).to(vector), v_pris), dim=1)
                ), dim=0)

        def vectors_to_angles(joint_angles):
            return torch.cat(
                (
                    3.14 * torch.sigmoid(joint_angles[:self.n_revol]) - 3.14/2,
                    0.6 * torch.sigmoid(joint_angles[self.n_revol:]) - 0.3
                )
            )

        def angles_to_vectors(joint_angles):
            return torch.cat(
                (
                    inverse_sigmoid(
                        torch.clip(joint_angles[:self.n_revol], -3.14/2, 3.14/2) / 3.14 + 0.5),
                    inverse_sigmoid(
                        torch.clip(joint_angles[self.n_revol:], -0.3, 0.3) / 0.6 + 0.5)
                )
            )

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # screwsplatting
        self.screw_activation = screw_from_vector
        self.screw_confidence_activation = torch.sigmoid
        self.inverse_screw_confidence_activation = inverse_sigmoid
        self.part_indice_activation = torch.nn.Softmax(dim=1)
        self.joint_angle_activation = vectors_to_angles
        self.inverse_joint_angle_activation = angles_to_vectors

    def __init__(self, sh_degree, optimizer_type="default", n_screws=None):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # screwsplatting
        self._screws = torch.empty(0)
        self._part_indices = torch.empty(0)

        # number of screws
        if n_screws is not None:
            self.n_revol = n_screws[0]
            self.n_pris = n_screws[1]

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,

            # screwsplatting
            self._screws,
            self._screw_confs,
            self._part_indices,
            self._joint_angles,
            self.n_revol,
            self.n_pris
        )
    
    def restore(self, model_args, training_args=None):
        (
            self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,

            # screwsplatting
            self._screws,
            self._screw_confs,
            self._part_indices,
            self._joint_angles,
            self.n_revol,
            self.n_pris
            ) = model_args
        
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    @property
    def get_screws(self):
        return self.screw_activation(self._screws)

    @property
    def get_screw_confs(self):
        return self.screw_confidence_activation(self._screw_confs)

    @property
    def get_part_indices(self):
        return self.part_indice_activation(self._part_indices)

    @property
    def get_joint_angles(self):
        return [
            self.joint_angle_activation(joint_angle) for joint_angle in self._joint_angles]

    @property
    def get_n_gaussians(self):
        return len(self._xyz)

    @property
    def get_n_screws(self):
        return len(self._screws)

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        
        # lr scale
        self.spatial_lr_scale = spatial_lr_scale
        
        # initialize point clouds and features
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # initialize scales (sphere with radius = min distance)
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # initialize rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # initialize opacities
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # convert to nn parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def create_screws_and_parts(self):
        
        # initialize screws with confidence
        screws = 1 * torch.rand(self.n_revol + self.n_pris, 6).float().cuda() - 0.5
        screw_confidences = self.inverse_screw_confidence_activation(
            0.9 * torch.ones(self.n_revol + self.n_pris, dtype=torch.float, device="cuda"))

        # initialize part indices (random, multiple)
        part_indices = torch.zeros(self._xyz.shape[0], len(screws)+1).float().cuda()

        # convert to nn parameters
        self._screws = nn.Parameter(screws.requires_grad_(True))
        self._screw_confs = nn.Parameter(screw_confidences.requires_grad_(True))
        self._part_indices = nn.Parameter(part_indices.requires_grad_(True))

    def create_joint_angles(self, cam_infos):

        # number of articulation
        num_arts = len(list(set([info.art_idx for info in cam_infos])))

        # initialize joint angles
        joint_angles = [
            torch.zeros(len(self._screws)).float().cuda() for _ in range(num_arts)]

        # convert to nn parameters
        self._joint_angles = [
            nn.Parameter(joint_angle.requires_grad_(True)) for joint_angle in joint_angles]

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # gaussians optimizer
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._part_indices], 'lr': training_args.part_index_lr, "name": "part_indices"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # screw parameters optimizer
        l = [
            {'params': [self._screws], 'lr': training_args.screw_lr, "name": "screw"}
        ]
        self.screw_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # screw confidence optimizer
        l = [
            {'params': [self._screw_confs], 'lr': training_args.screw_confidence_lr, "name": "screw_confs"}
        ]
        self.screw_confidence_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # exposure optimizer
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # joint angle optimizer
        self.joint_angle_optimizer = [
            torch.optim.Adam(
                [{'params': [joint_angle], 'lr': training_args.joint_angle_lr, "name": f"joint_angle_{i}"}], lr=0.0, eps=1e-15) 
            for i, joint_angle in enumerate(self._joint_angles)]

        # xyz scheduler
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        
        # exposure scheduler
        self.exposure_scheduler_args = get_expon_lr_func(
            lr_init=training_args.exposure_lr_init, 
            lr_final=training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_screw_conf_part_and_joint_angle(self):

        # get parameters
        screws = self.get_screws
        screw_confs = self.get_screw_confs
        part_indices = self.get_part_indices
        part_indices_int = torch.argmax(part_indices, dim=1)
        xyz = self.get_xyz
        rotation = self.get_rotation
        joint_angle_list = self.get_joint_angles
        num_arts = len(joint_angle_list)
        if torch.sum(screw_confs > 0.1) > 0:
            mean_joint_angle = joint_angle_list[
                torch.mode(
                    torch.stack(joint_angle_list)[:, screw_confs > 0.1].median(0)[1]
                )[0]
            ]
        else:
            mean_joint_angle = joint_angle_list[randint(0, num_arts - 1)]

        # gaussian poses
        screws = screws * mean_joint_angle.unsqueeze(1) # [N_s, 6]
        screw_transforms = exp_se3(screws) # [N_s, 4, 4]
        screw_transforms = torch.cat(
            (torch.eye(4).unsqueeze(0).to(screw_transforms), screw_transforms), 
            dim=0
        ) # [N_s+1, 4, 4]
        screw_transforms = screw_transforms[part_indices_int] # [N, 4, 4]
        screw_rotations = screw_transforms[:, :3, :3] # [N, 3, 3]
        screw_translations = screw_transforms[:, :3, 3] # [N, 3]
        xyz_new = (
            screw_rotations @ xyz.unsqueeze(-1) + screw_translations.unsqueeze(-1)
        ).squeeze(-1) # [N, 3]
        rotation_new = matrix_to_quaternion(
            screw_rotations @ quaternion_to_matrix(rotation)
        ) # [N, 4]
        optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "xyz")
        self._xyz = optimizable_tensors["xyz"]     
        optimizable_tensors = self.replace_tensor_to_optimizer(rotation_new, "rotation")
        self._rotation = optimizable_tensors["rotation"]     
        
        # part indices
        part_indices_new = torch.zeros(part_indices.shape[0], len(screw_confs)+1).float().cuda()
        optimizable_tensors = self.replace_tensor_to_optimizer(part_indices_new, "part_indices")
        self._part_indices = optimizable_tensors["part_indices"]         

        # screw confidence
        screw_confs_new = self.inverse_screw_confidence_activation(
            torch.max(screw_confs, torch.ones_like(self.get_screw_confs)*0.9))
        optimizable_tensors = self.replace_tensor_to_screw_conf_optimizer(screw_confs_new, "screw_confs")
        self._screw_confs = optimizable_tensors["screw_confs"] 

        # joint angles
        joint_angle_list_new = [
            self.inverse_joint_angle_activation(joint_angle_list[i] - mean_joint_angle) for i in range(num_arts)]
        optimizable_tensors_list = self.replace_tensor_to_joint_angle_optimizer(
            joint_angle_list_new)
        self._joint_angles = [
            optimizable_tensors[f"joint_angle_{i}"] for i, optimizable_tensors in enumerate(optimizable_tensors_list)]

    def prune_screws(self, screw_conf_thres, joint_bound_thres, radii):

        # get parameters
        screws = self.get_screws
        screw_confs = self.get_screw_confs
        part_indices = self.get_part_indices
        part_indices_int = torch.argmax(part_indices, dim=1)
        joint_angle_list = self.get_joint_angles
        num_arts = len(joint_angle_list)

        # prune low confidence screws
        deactivated_screws_int = torch.where(screw_confs <= screw_conf_thres)[0]
        activated_screws_bool = (screw_confs > screw_conf_thres)
        activated_parts_bool = torch.cat(
            (torch.tensor([True]).to(activated_screws_bool), activated_screws_bool)
        )
        deactivated_gaussians_bool = torch.isin(part_indices_int - 1, deactivated_screws_int)

        # prune screws
        screws_new = screws[activated_screws_bool]
        _screws_new = self._screws[activated_screws_bool]
        screw_confs_new = screw_confs[activated_screws_bool]
        _screw_confs_new = self._screw_confs[activated_screws_bool]
        _part_indices_new = self._part_indices[~deactivated_gaussians_bool][:, activated_parts_bool]
        joint_angle_list_new = [joint_angle_list[i][activated_screws_bool] for i in range(num_arts)]

        # prune gaussians
        self.prune_points(deactivated_gaussians_bool, skip_temp_radii=True)
        optimizable_tensors = self.replace_tensor_to_optimizer(_part_indices_new, "part_indices")
        self._part_indices = optimizable_tensors["part_indices"]   

        # num screws
        self.n_pris = self.n_pris - (deactivated_screws_int >= self.n_revol).sum().item()
        self.n_revol = self.n_revol - (deactivated_screws_int < self.n_revol).sum().item()
        joint_bound_thres = torch.cat([
            torch.full((self.n_revol,), joint_bound_thres[0]).to(screws),
            torch.full((self.n_pris,), joint_bound_thres[1]).to(screws)])

        # prune low joint bound screws
        part_indices_new = self.get_part_indices
        part_indices_int_new = torch.argmax(part_indices_new, dim=1)
        lower_limit = torch.min(torch.stack(joint_angle_list_new), dim=0)[0]
        upper_limit = torch.max(torch.stack(joint_angle_list_new), dim=0)[0]
        joint_bounds = upper_limit - lower_limit
        median_joint_angle = torch.stack(joint_angle_list_new).median(0)[1]
        invalid_screws_int = torch.where(joint_bounds <= joint_bound_thres)[0]
        valid_screws_bool = (joint_bounds > joint_bound_thres)
        valid_parts_bool = torch.cat(
            (torch.tensor([True]).to(valid_screws_bool), valid_screws_bool)
        )
        invalid_gaussians_bool= torch.isin(part_indices_int_new - 1, invalid_screws_int)

        # transform gaussians
        xyz_new = self.get_xyz
        rotation_new = self.get_rotation
        opacity_new = self.get_opacity
        screws_new = screws_new * median_joint_angle.unsqueeze(1) # [N_s, 6]
        screw_transforms = exp_se3(screws_new) # [N_s, 4, 4]
        screw_transforms = torch.cat(
            (torch.eye(4).unsqueeze(0).to(screw_transforms), screw_transforms), 
            dim=0
        ) # [N_s+1, 4, 4]
        screw_transforms = screw_transforms[part_indices_int_new[invalid_gaussians_bool] - 1] # [N, 4, 4]
        screw_rotations = screw_transforms[:, :3, :3] # [N, 3, 3]
        screw_translations = screw_transforms[:, :3, 3] # [N, 3]
        xyz_new[invalid_gaussians_bool, :] = (
            screw_rotations @ xyz_new[invalid_gaussians_bool, :].unsqueeze(-1) + screw_translations.unsqueeze(-1)
        ).squeeze(-1) # [N, 3]
        rotation_new[invalid_gaussians_bool, :] = matrix_to_quaternion(
            screw_rotations @ quaternion_to_matrix(rotation_new[invalid_gaussians_bool, :])
        ) # [N, 4]
        opacity_new[invalid_gaussians_bool, 0] = (
            opacity_new[invalid_gaussians_bool, 0]
            * screw_confs_new[part_indices_int_new[invalid_gaussians_bool] - 1]
        )
        _opacity_new = self.inverse_opacity_activation(opacity_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "xyz")
        self._xyz = optimizable_tensors["xyz"]     
        optimizable_tensors = self.replace_tensor_to_optimizer(rotation_new, "rotation")
        self._rotation = optimizable_tensors["rotation"]     
        optimizable_tensors = self.replace_tensor_to_optimizer(_opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]     

        # modift part indices
        switching_part_indices = torch.zeros_like(part_indices_int_new)
        switching_part_indices[invalid_gaussians_bool] = part_indices_int_new[invalid_gaussians_bool]
        _part_indices_new_2 = _part_indices_new.clone()
        _part_indices_new[torch.arange(_part_indices_new_2.shape[0]), 0] = _part_indices_new_2[torch.arange(_part_indices_new_2.shape[0]), switching_part_indices]
        _part_indices_new[torch.arange(_part_indices_new_2.shape[0]), switching_part_indices] = _part_indices_new_2[torch.arange(_part_indices_new_2.shape[0]), 0]

        # prune screws
        _screws_new = _screws_new[valid_screws_bool]
        _screw_confs_new = _screw_confs_new[valid_screws_bool]
        _part_indices_new = _part_indices_new[:, valid_parts_bool]
        joint_angle_list_new = [joint_angle_list_new[i][valid_screws_bool] for i in range(num_arts)]  

        # prune screws
        optimizable_tensors = self.replace_tensor_to_optimizer(_part_indices_new, "part_indices")
        self._part_indices = optimizable_tensors["part_indices"]  
        optimizable_tensors = self.replace_tensor_to_screw_optimizer(_screws_new, "screw")
        self._screws = optimizable_tensors["screw"] 
        optimizable_tensors = self.replace_tensor_to_screw_conf_optimizer(_screw_confs_new, "screw_confs")
        self._screw_confs = optimizable_tensors["screw_confs"] 

        # num screws
        self.n_pris = self.n_pris - (invalid_screws_int >= self.n_revol).sum().item()
        self.n_revol = self.n_revol - (invalid_screws_int < self.n_revol).sum().item()

        # joint angles
        joint_angle_list_new = [
            self.inverse_joint_angle_activation(joint_angle_list_new[i]) for i in range(num_arts)]
        optimizable_tensors_list = self.replace_tensor_to_joint_angle_optimizer(
            joint_angle_list_new)
        self._joint_angles = [
            optimizable_tensors[f"joint_angle_{i}"] for i, optimizable_tensors in enumerate(optimizable_tensors_list)]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensor_to_screw_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.screw_optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.screw_optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.screw_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.screw_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensor_to_screw_conf_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.screw_confidence_optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.screw_confidence_optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.screw_confidence_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.screw_confidence_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensor_to_joint_angle_optimizer(self, tensor_list):
        optimizable_tensors_list = []
        for i, tensor in enumerate(tensor_list):
            optimizable_tensors = {}
            for group in self.joint_angle_optimizer[i].param_groups:
                stored_state = self.joint_angle_optimizer[i].state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.joint_angle_optimizer[i].state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.joint_angle_optimizer[i].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            optimizable_tensors_list.append(optimizable_tensors)
        return optimizable_tensors_list


    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, skip_temp_radii=False):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._part_indices = optimizable_tensors["part_indices"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if not skip_temp_radii:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
            self, new_xyz, new_features_dc, new_features_rest, 
            new_opacities, new_scaling, new_rotation, new_tmp_radii, 
            new_part_indices):

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "part_indices": new_part_indices}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._part_indices = optimizable_tensors["part_indices"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # screwsplatting
        new_part_indices = self._part_indices[selected_pts_mask].repeat(N,1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, 
            new_scaling, new_rotation, new_tmp_radii, new_part_indices)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # screwsplatting
        new_part_indices = self._part_indices[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, 
            new_scaling, new_rotation, new_tmp_radii, new_part_indices)

    def densify_and_prune(
            self, max_grad, min_opacity, extent, 
            max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # prune indices
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # prune points
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height):
        """
        Customized for gsplat, edit here back to Inria if you want t o go back
        """

        grad = viewspace_point_tensor.grad.squeeze(0) # [N*(N_s+1), 2]
        grad = grad.reshape(self.get_n_gaussians, -1, 2) # [N, N_s+1, 2]

        # Normalize the gradient to [-1, 1] screen size
        grad[:, :, 0] *= width * 0.5
        grad[:, :, 1] *= height * 0.5

        self.xyz_gradient_accum[update_filter] += torch.sum(
            torch.norm(grad[update_filter, :, :2], dim=-1, keepdim=True),
            dim=1
        )
        self.denom[update_filter] += 1

#############################################################
################## ORIGINAL GAUSSIAN MODEL ##################
#############################################################
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args=None):
        (
            self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        
        # lr scale
        self.spatial_lr_scale = spatial_lr_scale
        
        # initialize point clouds and features
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # initialize scales (sphere with radius = min distance)
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # initialize rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # initialize opacities
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # convert to nn parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # gaussians optimizer
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # exposure optimizer
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # xyz scheduler
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

        # exposure scheduler
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init, 
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, 
        new_opacities, new_scaling, new_rotation, new_tmp_radii):
        
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # prune indices
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # prune points
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height):
        """
        Dr. Robot: customized for gsplat, edit here back to Inria if you want t o go back
        """

        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]

        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5

        # gradient accumulate
        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
   