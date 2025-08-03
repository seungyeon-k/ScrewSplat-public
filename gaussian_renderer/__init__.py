import torch
import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel, ScrewGaussianModel
from articulated_object.utils import exp_se3
from articulated_object.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion

#############################################################
#################### SCREWSPLAT RENDERER ####################
#############################################################
def render_with_screw(viewpoint_camera, 
           pc : ScrewGaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           separate_sh = False, 
           override_color = None, 
           use_trained_exp = False,
           desired_joint_angle = None,
           activate_screw_thres = None,
           render_mode="RGB"
           ):

    # Set up camera parameters
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.cx],
            [0, focal_length_y, viewpoint_camera.cy],
            [0, 0, 1],
        ],
        device="cuda",
    ).float()
    art_idx = viewpoint_camera.art_idx

    # get gaussian params
    means3D = pc.get_xyz # [N, 3]
    opacity = pc.get_opacity # [N, 1]
    n_gaussians = means3D.shape[0]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc.get_scaling * scaling_modifier # [N, 3]
    rotations = pc.get_rotation # [N, 4]

    # get screw informations
    screws = pc.get_screws # [N_s, 6]
    screw_confs = pc.get_screw_confs # [N_s, ]
    part_indices = pc.get_part_indices # [N, N_s+1]
    
    # joint angles
    if desired_joint_angle is not None: # for inference
        joint_angle = desired_joint_angle
    else: # for training
        joint_angle = pc.get_joint_angles[art_idx] # [N_s,]

    # activated indices
    if activate_screw_thres is not None:
        activated_screw_indices = torch.where(screw_confs > activate_screw_thres)[0]
        screws = screws[activated_screw_indices]
        screw_confs = screw_confs[activated_screw_indices]
        part_indices = part_indices[
            :, torch.cat((torch.tensor([0]).to(activated_screw_indices), activated_screw_indices + 1))]
        joint_angle = joint_angle[activated_screw_indices]
    else:
        activated_screw_indices = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree
    n_features = shs.shape[1]

    # screw transforms
    n_screws = screws.shape[0]
    screws = screws * joint_angle.unsqueeze(1) # [N_s, 6]
    screw_transforms = exp_se3(screws) # [N_s, 4, 4]
    screw_transforms = torch.cat(
        (torch.eye(4).unsqueeze(0).to(screw_transforms), screw_transforms), 
        dim=0
    ) # [N_s+1, 4, 4]
    screw_rotations = screw_transforms[:, :3, :3] # [N_s+1, 3, 3]
    screw_translations = screw_transforms[:, :3, 3] # [N_s+1, 3]
    screw_confs = torch.cat(
        (torch.tensor([1.0]).to(screw_confs), screw_confs), 
        dim=0
    ) # [N_s+1, ]

    # augmented gaussians
    augmented_means3D = (
        screw_rotations.unsqueeze(0).repeat(n_gaussians, 1, 1, 1)
        @ means3D.unsqueeze(1).repeat(1, n_screws+1, 1).unsqueeze(-1)
        + screw_translations.unsqueeze(0).repeat(n_gaussians, 1, 1).unsqueeze(-1)
    ).squeeze(-1) # [N, N_s+1, 3]    
    augmented_opacity = (
        opacity.repeat(1, n_screws+1) 
        * part_indices 
        * screw_confs.unsqueeze(0).repeat(n_gaussians, 1)
    )# [N, N_s+1]
    augmented_scales = scales.unsqueeze(1).repeat(1, n_screws+1, 1) # [N, N_s+1, 3]
    augmented_screw_rotations = screw_rotations.unsqueeze(0).repeat(n_gaussians, 1, 1, 1) # [N, N_s+1, 3, 3]
    augmented_rotations = matrix_to_quaternion(
        augmented_screw_rotations
        @ quaternion_to_matrix(rotations).unsqueeze(1).repeat(1, n_screws+1, 1, 1)
    ) # [N, N_s+1, 4]
    augmeneted_shs = shs.unsqueeze(1).repeat(1, n_screws+1, 1, 1) # [N, N_s+1, K, 3]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device) 
    render_colors, render_alphas, info = rasterization(
        means=augmented_means3D.reshape(-1, 3),  # [N*(N_s+1), 3]
        quats=augmented_rotations.reshape(-1, 4),  # [N*(N_s+1), 4]
        scales=augmented_scales.reshape(-1, 3),  # [N*(N_s+1), 3]
        opacities=augmented_opacity.reshape(-1),  # [N*(N_s+1),]
        colors=augmeneted_shs.reshape(-1, n_features, 3), # [N*(N_s+1), K, 3]
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode=render_mode
    )

    # process outputs
    rendered_image = render_colors[0].permute(2, 0, 1) # [1, H, W, 3] -> [3, H, W]
    
    # process radii (version issue)
    radii = info["radii"].squeeze(0) # [N,] or [N, 2]
    if len(radii.shape) == 2 and radii.shape[1] == 2:
        radii = torch.max(radii, dim=1)[0] # [N,]
    elif len(radii.shape) == 1:
        pass
    else:
        raise ValueError('invalid radii data type')

    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # return
    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "activated_screw_indices": activated_screw_indices,
        "radii": radii,
        "art_idx": art_idx,
        "n_gaussians": n_gaussians,
        "n_screws": n_screws}

#############################################################
###################### GAUSSIAN RENDERER ####################
#############################################################
def render(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           separate_sh = False, 
           override_color = None, 
           use_trained_exp = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up camera parameters
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.cx],
            [0, focal_length_y, viewpoint_camera.cy],
            [0, 0, 1],
        ],
        device="cuda",
    ).float()

    # get gaussian params
    means3D = pc.get_xyz # [N, 3]
    opacity = pc.get_opacity # [N, 1]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc.get_scaling * scaling_modifier # [N, 3]
    rotations = pc.get_rotation # [N, 4]
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device) 
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=shs,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
    )

    # process outputs
    rendered_image = render_colors[0].permute(2, 0, 1) # [1, H, W, 3] -> [3, H, W]

    # process radii (gsplat version issue)
    radii = info["radii"].squeeze(0) # [N,] or [N, 2]
    if len(radii.shape) == 2 and radii.shape[1] == 2:
        radii = torch.max(radii, dim=1)[0] # [N,]
    elif len(radii.shape) == 1:
        pass
    else:
        raise ValueError('invalid radii data type')

    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # return
    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "radii": radii}

