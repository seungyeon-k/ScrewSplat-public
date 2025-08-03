import os
import torch
import random
from random import gauss, randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_with_screw
import sys
from scene import Scene, ArticulatedScene, ScrewGaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from copy import deepcopy
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

SPARSE_ADAM_AVAILABLE = False


def training_total(
        dataset, 
        opt, 
        pipe, 
        testing_iterations, 
        saving_iterations, 
        checkpoint_iterations, 
        checkpoint, 
        debug_from):

    tb_writer = prepare_output_and_logger(dataset, weight=opt.parsimony_weight_init)
    gaussians = ScrewGaussianModel(
        dataset.sh_degree, opt.optimizer_type, opt.n_screws)
    scene = ArticulatedScene(dataset, gaussians)

    # train screw gaussian
    print("============ Training Screw Gaussian ============")
    training(
        gaussians, scene, tb_writer, dataset, opt, pipe, 
        testing_iterations, saving_iterations, 
        checkpoint_iterations, checkpoint, 
        debug_from)

def training(gaussians, 
             scene,
             tb_writer,
             dataset,
             opt,
             pipe, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from):

    # initialize
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # viewpoints
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # mask value
    masks = torch.stack([c.mask for c in viewpoint_stack])
    mask_ratio = torch.sum(masks) / (masks.shape[0] * masks.shape[1] * masks.shape[2])
    mask_ratio = mask_ratio.detach().item()

    # for accurate gpu time measurement
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # weight schedulers
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, 
        opt.depth_l1_weight_final, 
        max_steps=opt.iterations)
    parsimony_weight = get_expon_lr_func(
        opt.parsimony_weight_init,
        opt.parsimony_weight_final,
        max_steps=opt.parsimony_weight_max_steps)

    # initialize for exponential moving average
    ema_recon_for_log = 0.0
    ema_parsimony_for_log = 0.0

    # progress bar
    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # iteration
    for iteration in range(first_iter, opt.iterations + 1):

        # iteration starts
        iter_start.record()
        
        # update lr according to iteration
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # background
        if opt.random_background:
            bg = torch.rand((3), device="cuda")
        else:
            bg = background

        # render
        render_pkg = render_with_screw(
            viewpoint_cam, gaussians, pipe, bg, 
            use_trained_exp=dataset.train_test_exp, 
            separate_sh=SPARSE_ADAM_AVAILABLE,
            activate_screw_thres=opt.activate_screw_thres)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        # visibility_filter = render_pkg["visibility_filter"]
        activated_screw_indices = render_pkg["activated_screw_indices"]
        radii = render_pkg["radii"]
        art_idx = render_pkg["art_idx"]
        n_gaussians = render_pkg["n_gaussians"]
        n_screws = render_pkg["n_screws"]

        # mask image
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # RGB loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
            opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs(
                (invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0     

        # parsimony loss
        if parsimony_weight(iteration) > 0:
            screw_confs = gaussians.get_screw_confs
            eps = 1e-10
            parsimony_loss = torch.mean(
                torch.sqrt(screw_confs + eps)
            )
            loss += parsimony_weight(iteration) * parsimony_loss
        else:
            parsimony_loss = torch.tensor([0.0])

        # backward
        loss.backward()

        # iteration end
        iter_end.record()

        with torch.no_grad():
            
            # exponential moving average
            ema_recon_for_log = 0.4 * (loss.item() - parsimony_weight(iteration) * parsimony_loss.item()) + 0.6 * ema_recon_for_log
            ema_parsimony_for_log = 0.4 * parsimony_loss.item() + 0.6 * ema_parsimony_for_log

            # progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Recon Loss": f"{ema_recon_for_log:.{5}f}", 
                    "Parsimony Loss": f"{ema_parsimony_for_log:.{5}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, iteration, Ll1, loss, parsimony_loss, parsimony_weight(iteration), l1_loss, 
                iter_start.elapsed_time(iter_end), 
                testing_iterations, scene, render_with_screw, 
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Max memory used: {mem:.2f} GB")
                scene.save(iteration)

            # Densification of gaussians
            if iteration < opt.densify_until_iter:

                # reshape
                if activated_screw_indices is not None:
                    radii = radii.reshape(n_gaussians, -1)
                    part_indices = gaussians.get_part_indices[
                        :, torch.cat((torch.tensor([0]).to(activated_screw_indices), activated_screw_indices + 1))]
                    radii = (radii.to(part_indices) * part_indices).sum(dim=1).to(radii)
                    visibility_filter = (radii > 0)

                else:
                    radii = radii.reshape(n_gaussians, n_screws+1)
                    part_indices = gaussians.get_part_indices
                    radii = (radii.to(part_indices) * part_indices).sum(dim=1).to(radii)
                    visibility_filter = (radii > 0)

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, 
                    image.shape[2], image.shape[1])

                # densify and prune gaussians
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # print(f'image name for densification: {viewpoint_cam.image_name}')
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, 
                        size_threshold, radii)
                
                # reset opacity
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Densification of screws and part labels
            if (sum(opt.n_screws) > 1) and (iteration < opt.densify_screws_until_iter):

                # reset screw confidence, part indices, and joint angles
                if (iteration - opt.opacity_reset_interval // 2) % opt.opacity_reset_interval == 0:
                    gaussians.reset_screw_conf_part_and_joint_angle()

            # prune screws
            if (sum(opt.n_screws) > 1) and (iteration == opt.prune_screws_iter):
                gaussians.prune_screws(
                    opt.screw_conf_thres, opt.joint_bound_thres, radii)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if iteration < opt.prune_screws_iter:
                    gaussians.joint_angle_optimizer[art_idx].step()
                    gaussians.joint_angle_optimizer[art_idx].zero_grad(set_to_none = True)
                    gaussians.screw_optimizer.step()
                    gaussians.screw_optimizer.zero_grad(set_to_none = True)
                    if sum(opt.n_screws) > 1:
                        gaussians.screw_confidence_optimizer.step()
                        gaussians.screw_confidence_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                # print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, weight=None):
    if not args.model_path:

        # get object name (unstable)
        # object_name = '/'.join(args.source_path.split('/')[-3:])
        list_object_name = args.source_path.split(os.sep)[-4:]
        if weight is None:
            object_name = os.sep.join(list_object_name)
        else:
            list_object_name[-1] = list_object_name[-1] + '_' + str(weight)
            object_name = os.sep.join(list_object_name)

        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(
            "./output/", object_name, unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
        tb_writer, iteration, Ll1, loss, parsimony_loss, parsimony_weight, l1_loss, elapsed, 
        testing_iterations, scene : ArticulatedScene, renderFunc, 
        renderArgs, train_test_exp):
    
    # loss report
    if tb_writer:
        tb_writer.add_scalar('train/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train/recon_loss', loss.item() - parsimony_weight * parsimony_loss.item(), iteration)
        tb_writer.add_scalar('train/parsimony_loss', parsimony_loss.item(), iteration)
        tb_writer.add_scalar('train/weighted_parsimony_loss', parsimony_weight * parsimony_loss.item(), iteration)
        tb_writer.add_scalar('train/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:

        # empty cache
        torch.cuda.empty_cache()
        
        # cameras
        train_cameras = scene.getTrainCameras()
        art_indices = list(set([info.art_idx for info in train_cameras]))
        train_cameras_list = [
            [info for info in train_cameras if info.art_idx == art_idx] for art_idx in art_indices
        ]

        # report articulation-wise
        train_cameras_list_for_art = []
        for art_idx, train_cameras in enumerate(train_cameras_list):
            
            # initialize metrics
            l1_test = 0.0
            psnr_test = 0.0

            # select indices
            train_cameras = [
                train_cameras[idx] for idx in range(0, len(train_cameras), 7)]
            train_cameras_list_for_art += train_cameras

            # iterate cameras
            for idx, viewpoint in enumerate(train_cameras):
                
                # images
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                
                # adjust resolution
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                
                # log images
                if tb_writer and (idx < 4):
                    tb_writer.add_images(
                        f'{viewpoint.image_name}/render', 
                        image[None], 
                        global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            f'{viewpoint.image_name}/gt', 
                            gt_image[None], 
                            global_step=iteration)
                
                # update metrics
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            
            # mean metrics
            psnr_test /= len(train_cameras)
            l1_test /= len(train_cameras)

            # log metrics
            print(f"[ITER {iteration}] Evaluating {art_idx+1}'th Articulation: L1 {l1_test} PSNR {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar('train_loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar('train_loss_viewpoint - psnr', psnr_test, iteration)

        # point information
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('gaussian/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            part_indices = scene.gaussians.get_part_indices
            for part_idx in range(part_indices.shape[1]):
                tb_writer.add_histogram(f"scene/{part_idx}th_label_histogram", part_indices[:, part_idx], iteration)

        # print screws
        screws = scene.gaussians.get_screws
        screw_confs = scene.gaussians.get_screw_confs
        screw_grads = torch.norm(scene.gaussians._screws.grad, dim=1)
        joint_angles = torch.stack(scene.gaussians.get_joint_angles)
        lower_limits = torch.min(joint_angles, dim=0)[0]
        upper_limits = torch.max(joint_angles, dim=0)[0]

        for screw_idx in range(screws.shape[0]):
            
            screw = screws[screw_idx].cpu().detach().tolist()
            screw = [round(num, 3) for num in screw]
            lower_limit = round(lower_limits[screw_idx].item(), 5)
            upper_limit = round(upper_limits[screw_idx].item(), 5)
            # print(f"[ITER {iteration}] Estimated {screw_idx+1}th Screw with Confidence {round(screw_confs[screw_idx].item(), 3)}: {screw}")
            # print(f"[ITER {iteration}] Estimated {screw_idx+1}th Screw with Confidence {round(screw_confs[screw_idx].item(), 3)} and Gradient {screw_grads[screw_idx]}")
            print(f"[ITER {iteration}] Estimated {screw_idx+1}th Screw with Confidence {round(screw_confs[screw_idx].item(), 3)} and Joint limit [{lower_limit}, {upper_limit}]")
        
        # print joint angles
        joint_angles = scene.gaussians.get_joint_angles
        for idx, j in enumerate(joint_angles):
            joint_angle = j.cpu().detach().tolist()
            joint_angle = [round(num, 3) for num in joint_angle]
            # print(f"[ITER {iteration}] Estimated {idx+1}th Articulation's Joint Angles: {joint_angle}")
        
        # report articulated object
        steps = 100
        n_joints = len(joint_angles)
        joint_limits = torch.tensor([[-3.14, 3.14]]).repeat(1, n_joints).float().cuda()
        weights = torch.linspace(
            0, 1, steps, 
            device="cuda", 
            dtype=joint_limits.dtype
        ).view(-1, *[1]*joint_limits[:, 0].dim())
        thetas = joint_limits[:, 0] + weights * (joint_limits[:, 1] - joint_limits[:, 0])

        # get random viewpoint
        cameras = train_cameras_list_for_art
        random.shuffle(cameras)

        # report articulation
        for i, camera in enumerate(cameras):
            for j, theta in enumerate(thetas):

                # images
                print(theta)
                renderArgs_ = renderArgs + (theta,)
                image = torch.clamp(
                    renderFunc(
                        camera, scene.gaussians,
                        *renderArgs_)["render"], 
                    0.0, 1.0)
                
                # adjust resolution
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                
                # log images
                if tb_writer and (i < 4):
                    tb_writer.add_images(
                        f'_articulation_iter_{iteration}/{camera.image_name.split("/")[-1]}',
                        image[None], 
                        global_step=j)

        # empty
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 70_000, 100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 70_000, 100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, 
        default=list(range(4999, 30000, 5000)))
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # set torch device globally
    torch.cuda.set_device(args.device)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_total(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from)

    # All done
    print("\nTraining complete.")
