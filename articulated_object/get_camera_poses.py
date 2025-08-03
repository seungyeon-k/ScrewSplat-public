import torch
import numpy as np
from articulated_object.utils import exp_se3

def get_camera_poses(
        num_phi=3, 
        phi_range=[20/180*np.pi, 80/180*np.pi],
        num_theta=3, 
        theta_range=[-np.pi, np.pi],
        radius=1.2,
        cam_center=[0.0, 0.0, 0.0]):
            
    # camera pose initialize
    view_poses = []
    
    # linspace
    rotating_phis = np.linspace(
        start = phi_range[0], stop = phi_range[1], num=num_phi)
    if theta_range[1] - theta_range[0] > 6.28: # full view
        rotating_thetas = np.linspace(
            start = theta_range[0], stop = theta_range[1], num=num_theta+1)[:-1]
    else:
        rotating_thetas = np.linspace(
            start = theta_range[0], stop = theta_range[1], num=num_theta)        

    # shelf params
    center = np.array([0, 0, 0])
    z_axis = np.array([0, 0, 1])
    v = - np.cross(z_axis, center)
    screw = np.concatenate((z_axis, v))

    # camera poses
    for phi in rotating_phis:
        
        # reference ee pose
        view_pose_init = np.eye(4)
        view_pose_init[:3, :3] = np.array([
            [0, -np.sin(phi), np.cos(phi)], 
            [-1, 0			 , 0],
            [0, -np.cos(phi), -np.sin(phi)]]
        )
        view_pose_init[:3, 3] = np.array(
            [-radius*np.cos(phi), 0, radius*np.sin(phi)])

        # rotating theta
        for theta in rotating_thetas:
            rotating_SE3 = exp_se3(torch.tensor(theta * screw)).numpy()
            view_pose = rotating_SE3.dot(view_pose_init)
            
            # translate
            view_pose[:3, 3] = view_pose[:3, 3] + np.array(cam_center)

            # append
            view_poses.append(view_pose)

    return view_poses