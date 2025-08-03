import torch
import os
from argparse import ArgumentParser
from arguments import get_combined_args
from gaussian_renderer import ScrewGaussianModel
import numpy as np
from gaussian_renderer import render_with_screw
from scene import ScrewGaussianModel
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2
import matplotlib.pyplot as plt
from copy import deepcopy
from skopt import gp_minimize
from utils.loss_utils import l1_loss, ssim
from functools import partial

class MiniCam(NamedTuple):
    FoVx: np.array
    FoVy: np.array
    image_width: int
    image_height: int
    cx: np.array
    cy: np.array
    art_idx: int
    world_view_transform: torch.tensor

class Image2JointAngles:

    def __init__(self, args, joint_angle=None):
        
        # device
        self.device = "cuda"

        # process
        self.process_data(args)
    
        # initial joint angle
        if joint_angle is None:
            self.init_joint_angles = self.gaussians.get_joint_angles[-1].to(self.device)
        else:
            self.init_joint_angles = joint_angle.to(self.device)

    def get_image_loss(self, joint_angles, images, weight=0.2):
        rendered_images = []
        for view_idx in range(self.n_views):
            image = self.render(joint_angles, view_idx)
            rendered_images.append(image)
        rendered_images = torch.stack(rendered_images)
        
        Ll1 = l1_loss(rendered_images, images)
        ssim_value = ssim(rendered_images, images)
        loss = (1.0 - weight) * Ll1 + weight * (1.0 - ssim_value)        

        return loss
    
    def target_func_bayes(self, x, images):
        joint_angles = self.init_joint_angles.clone()
        joint_angles[self.valid_screw_mask] = torch.tensor(x).to(self.device)
        return self.get_image_loss(joint_angles, images).item()
    
    def bayes_optimize(self, images):

        # valid screws
        self.valid_screw_mask = self.gaussians.get_screw_confs > 0.1
        search_range = []
        for i in range(self.valid_screw_mask.sum()):
            search_range.append((self.joint_limits[self.valid_screw_mask, 0][i].item(), self.joint_limits[self.valid_screw_mask, 1][i].item()))
        
        # partial
        target_func = partial(self.target_func_bayes, images=images.to(self.device))

        # optimization
        result = gp_minimize(target_func,       
                  search_range,      
                  acq_func="EI",      
                  n_calls=15,         
                  n_random_starts=5,
                  verbose=False)
        
        # final joint angle
        out_joint_angles = self.init_joint_angles.clone()
        out_joint_angles[self.valid_screw_mask] = torch.tensor(result.x).to(self.device)
        
        return out_joint_angles
    
    def gradient_descent_optimize(self, images):
        
        lower = self.joint_limits[:, 0].detach()
        upper = self.joint_limits[:, 1].detach()
        # joint angles
        def x_to_angle(x):
            return torch.sigmoid(x) * (upper-lower) + lower
        
        x = torch.zeros_like(self.init_joint_angles).cuda().requires_grad_()
        optimizer = torch.optim.Adam([x], lr=0.01)

        for i in range(100):
            joint_angle = x_to_angle(x)
            loss = self.get_image_loss(joint_angle, images.to(self.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("iter : ", i, "loss : ", loss.item())
        
        return joint_angle

    def render(self, joint_angle, camera_idx):

        # render
        render_pkg = render_with_screw(
            self.cameras[camera_idx], self.gaussians, None, self.bg, 
            use_trained_exp=None, separate_sh=None, activate_screw_thres=None,
            desired_joint_angle=joint_angle,
            )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        return image # (3, 480, 640)

    def visualization(self, target_joint_angles, images, plot_idx=2):

        plt.subplot(1, 3, 1)
        result_image = self.render(self.init_joint_angles, plot_idx)
        plt.imshow(result_image.permute(1, 2, 0).detach().cpu().numpy())
        plt.title("latest")
        plt.subplot(1, 3, 2)
        result_image = images[plot_idx]
        plt.imshow(result_image.permute(1, 2, 0).detach().cpu().numpy())
        plt.title("current")
        plt.subplot(1, 3, 3)
        result_image = self.render(target_joint_angles, plot_idx)
        plt.imshow(result_image.permute(1, 2, 0).detach().cpu().numpy())
        plt.title("estimated")
        plt.show()

    def process_data(self, args):

        # process args
        source_path = args.source_path
        #temp#
        source_path = os.path.join(*source_path.split('/')[4:])
        ######
        model_path = args.model_path
        images = args.images
        depths = args.depths
        white_background = args.white_background
        sh_degree = args.sh_degree

        # background color
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # get recent checkpoint
        checkpoint_name_list = [file for file in os.listdir(model_path) if file.endswith('.pth')]
        sorted_checkpoints_names = sorted(checkpoint_name_list, key=lambda x: int(x.replace('chkpnt', '').replace('.pth', '')))
        checkpoint_name = sorted_checkpoints_names[-1]

        # initialize gaussian        
        self.gaussians = ScrewGaussianModel(sh_degree)
        checkpoint = os.path.join(model_path, checkpoint_name)
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        self.gaussians.restore(model_params)

        # get joint limit
        joint_angles = self.gaussians.get_joint_angles
        lower_limit = torch.min(torch.stack(joint_angles), dim=0)[0]
        upper_limit = torch.max(torch.stack(joint_angles), dim=0)[0]
        self.joint_limits = torch.cat(
            (lower_limit.unsqueeze(-1), upper_limit.unsqueeze(-1)), dim=1)

        # load extrinsics and intrinsics
        art_name = '0'
        intrinsic = np.load(os.path.join(source_path, art_name, 'intrinsic.npy'))
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # load extrinsics
        extrinsic_names = os.listdir(os.path.join(source_path, art_name, 'extrinsics'))
        extrinsic_names.sort()
        self.n_views = len(extrinsic_names)

        # process camera info
        self.cameras = []
        for idx, name in enumerate(extrinsic_names):

            # get camera to world frame
            extrinsic = np.load(os.path.join(source_path, art_name, 'extrinsics', name))

            # get the world-to-camera transform and set R, T
            R = np.transpose(extrinsic[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = extrinsic[:3, 3]

            # RGB image
            image_name = f'{name.split(".")[0]}.png'
            image_path = os.path.join(source_path, art_name, 'images', image_name)
            image = Image.open(image_path)

            # fov
            FoVx = 2 * np.arctan(image.size[0] / (2 * fy))
            FoVy = 2 * np.arctan(image.size[1] / (2 * fx))

            # world view transform
            world_view_transform = torch.tensor(
                getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()

            self.cameras.append(
                MiniCam(
                    FoVx=FoVx, FoVy=FoVy, image_width=image.size[0],
                    image_height=image.size[1], cx=cx, cy=cy, art_idx=0,
                    world_view_transform=world_view_transform
                )
            )

    def get_test_images(self, args, source_path=None, selected_idx=0):
    
        # process args
        if source_path is None:
            source_path = args.source_path
            source_path = os.path.join(*source_path.split('/')[4:])

        # load extrinsics and intrinsics
        art_names = [name for name in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, name))]
        art_names.sort()

        # process camera info
        images_list = []
        for art_idx, art_name in enumerate(art_names):

            # extrinsic file names
            extrinsic_names = os.listdir(os.path.join(source_path, art_name, 'extrinsics'))
            extrinsic_names.sort()

            # list
            images = []

            for img_idx, name in enumerate(extrinsic_names):

                # RGB image
                image_name = f'{name.split(".")[0]}.png'
                image_path = os.path.join(source_path, art_name, 'images', image_name)
                image = Image.open(image_path)
                image = np.array(image.convert("RGBA"))

                # modify image
                bg = np.array([1, 1, 1])
                norm_data = image / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = torch.tensor(arr).float().permute(2, 0, 1)

                # append
                images.append(image)

            # append
            images = torch.stack(images)
            images_list.append(images)
        
        return images_list[selected_idx]

if __name__ == "__main__":

    # parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    args = get_combined_args(parser)

    # model load
    model = Image2JointAngles(args)