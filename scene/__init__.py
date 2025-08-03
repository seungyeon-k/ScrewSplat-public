import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, ScrewGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

#############################################################
################# ARTICULATED 3D GAUSSIANS ##################
#############################################################
class ArticulatedScene:
    gaussians : ScrewGaussianModel

    def __init__(self, 
            args : ModelParams, 
            gaussians : ScrewGaussianModel, 
            load_iteration=None, 
            shuffle=True, 
            resolution_scales=[1.0]):
        
        """b
        :param path: Path to colmap scene main folder.
        """
        
        # initialize
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # initialize cameras
        self.train_cameras = {}
        self.test_cameras = {}

        # scene info
        scene_info = sceneLoadTypeCallbacks["ArtPartnet"](
            args.source_path, args.white_background)

        # joint limits
        self.joint_limits = scene_info.joint_limits

        # process camera
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # # camera shuffle
        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)
        #     random.shuffle(scene_info.test_cameras)

        # nerf normalization (spatial learning weight)
        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_extent = 2

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # load gaussian
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"), 
                args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, 
                scene_info.train_cameras, 
                self.cameras_extent)

        # load screws and part indices
        if self.loaded_iter:
            raise NotImplementedError
        else:
            self.gaussians.create_screws_and_parts()

        # load joint angles
        if self.loaded_iter:
            raise NotImplementedError
        else:
            self.gaussians.create_joint_angles(
                scene_info.train_cameras)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


#############################################################
################### ORIGINAL 3D GAUSSIANS ###################
#############################################################
class Scene:
    gaussians : GaussianModel

    def __init__(self, 
            args : ModelParams, 
            gaussians : GaussianModel, 
            load_iteration=None, 
            shuffle=True, 
            resolution_scales=[1.0]):

        # initialize
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.random_gaussian_init = True

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # initialize cameras
        self.train_cameras = {}
        self.test_cameras = {}

        # scene info
        scene_info = sceneLoadTypeCallbacks["Partnet"](
            args.source_path, args.white_background)

        # process camera (cameras.json) and point cloud (input.ply)
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # camera shuffle
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # nerf normalization (spatial learning weight)
        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_extent = 2

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # load points
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"),
                args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, 
                scene_info.train_cameras, 
                self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
