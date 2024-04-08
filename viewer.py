from copy import deepcopy
import os
import random
from typing import NamedTuple
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from PIL import Image
import json
#from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov, fov2focal
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
from kornia import create_meshgrid
import cv2

class ViewerCameraInfo:
    def __init__(self, R, T, FovY, FovX, width, height, timestamp=0.0, fl_x=-1.0, fl_y=-1.0, cx=-1.0, cy=-1.0):
        self.R = R
        self.T = T
        self.FoVy = FovY
        self.FoVx = FovX
        self.image_width = width
        self.image_height = height
        self.timestamp = timestamp
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0

        self.update(self.R, self.T)

    def update(self, R, T):
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        if self.cx > 0:
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, self.cx, self.cy, self.fl_x, self.fl_y, self.image_width, self.image_height).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_rays(self):
        grid = create_meshgrid(int(self.image_height), int(self.image_width), normalized_coordinates=False)[0] + 0.5
        i, j = grid.unbind(-1)
        pts_view = torch.stack([(i-self.cx)/self.fl_x, (j-self.cy)/self.fl_y, torch.ones_like(i), torch.ones_like(i)], -1).to(self.world_view_transform.device)
        c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))
        pts_world =  pts_view @ c2w.T
        directions = pts_world[...,:3] - self.camera_center[None,None,:]
        return self.camera_center[None,None], directions / torch.norm(directions, dim=-1, keepdim=True)

    def cuda(self):
        cuda_copy = deepcopy(self)
        for k, v in cuda_copy.__dict__.items():
            if isinstance(v, torch.Tensor):
                cuda_copy.__dict__[k] = v.to("cuda")
        return cuda_copy

'''
class ViewerCameraInfo(NamedTuple):
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0

    def cuda(self):
        cuda_copy = deepcopy(self)
        for k, v in cuda_copy.__dict__.items():
            if isinstance(v, torch.Tensor):
                cuda_copy.__dict__[k] = v.to("cuda")
        return cuda_copy
'''

def readCamerasFromTransforms(transformsfilePath, time_duration=None, frame_ratio=1):
    with open(transformsfilePath) as json_file:
        contents = json.load(json_file)
    if "camera_angle_x" in contents:
        fovx = contents["camera_angle_x"]
        
    width = contents["w"]
    height = contents["h"]
    frames = contents["frames"]
    idx_frame = frames[0]
    #idx = idx_frame[0]
    frame = idx_frame #idx_frame[1]
    timestamp = frame.get('time', 0.0)
    if frame_ratio > 1:
        timestamp /= frame_ratio
    if time_duration is not None and 'time' in frame:
        if timestamp < 0.0 or timestamp > time_duration:
            return

    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = np.array(frame["transform_matrix"])
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    
    if 'fl_x' in frame and 'fl_y' in frame and 'cx' in frame and 'cy' in frame:
        FovX = FovY = -1.0
        fl_x = frame['fl_x']
        fl_y = frame['fl_y']
        cx = frame['cx']
        cy = frame['cy']
        return ViewerCameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, width=width, height=height, timestamp=timestamp,
                    fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
        
    elif 'fl_x' in contents and 'fl_y' in contents and 'cx' in contents and 'cy' in contents:
        FovX = FovY = -1.0
        fl_x = contents['fl_x']
        fl_y = contents['fl_y']
        cx = contents['cx']
        cy = contents['cy']
        return ViewerCameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, width=width, height=height, timestamp=timestamp,
                    fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
    else:
        fovy = focal2fov(fov2focal(fovx, width), height)
        FovY = fovy
        FovX = fovx
        return ViewerCameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, width=width, height=height, timestamp=timestamp)

def rotation_matrix(axis, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle_radians), -np.sin(angle_radians)],
                         [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        return np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                         [0, 1, 0],
                         [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        return np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                         [np.sin(angle_radians), np.cos(angle_radians), 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Unknown rotation axis")

def update_camera(key, camera):
    translation_speed = 0.1  # Adjust the speed as necessary
    rotation_speed = 5.0  # Adjust the angle in degrees as necessary

    # Get current R and T
    R = camera.R
    T = camera.T

    # Define the translation increments based on current rotation
    if key == ord('w'):  # Move down in the world (which appears as "up" in your camera)
        T -= np.array([0, 0, translation_speed])
    elif key == ord('s'):  # Move up in the world (which appears as "down" in your camera)
        T += np.array([0, 0, translation_speed])
    elif key == ord('a'):  # Move right in the world (which appears as "left" in your camera)
        T += np.array([translation_speed, 0, 0])
    elif key == ord('d'):  # Move forward in the world (which appears as "into the screen" in your camera)
        T -= np.array([translation_speed, 0, 0])
    elif key == ord('c'):  # Move backward in the world (which appears as "coming out of the screen" in your camera)
        T += np.array([0, translation_speed, 0])
    elif key == ord('z'):  # Move left in the world (which appears as "right" in your camera)
        T -= np.array([0, translation_speed, 0])


    # Define the rotation increments
    if key == 82:  # Up arrow - Pitch down
        R = R @ rotation_matrix('x', -rotation_speed)
    elif key == 84:  # Down arrow - Pitch up
        R = R @ rotation_matrix('x', rotation_speed)
    elif key == 81:  # Left arrow - Yaw left
        R = R @ rotation_matrix('y', -rotation_speed)
    elif key == 83:  # Right arrow - Yaw right
        R = R @ rotation_matrix('y', rotation_speed)

    # Reset rotation to identity matrix
    if key == ord('x'):  # Reset rotation
        R = np.eye(3)

    # Update the camera's R and T
    camera.update(R, T)

    return camera

def viewing(dataset, opt, pipe, checkpoint,
             gaussian_dim, time_duration, rot_4d, force_sh_3d, args):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    checkpoint_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, checkpoint_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Viewer: Loaded from checkpoint: {checkpoint_iter}")


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    #lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    #for lambda_name in lambda_all:
    #    vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
    else:
        env_map = None
        
    gaussians.env_map = env_map
        
    viewpoint_cam = readCamerasFromTransforms(args.transforms_path, gaussians.time_duration[1]) #batch_data[batch_idx]
    viewpoint_cam = viewpoint_cam.cuda()

    target_timestamp = 1.0
    viewpoint_cam.timestamp = target_timestamp

    


    with torch.no_grad():
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        img = render_pkg["render"].detach().clone().permute(1,2,0).cpu().numpy()

        cv2.namedWindow('Render', cv2.WINDOW_NORMAL)

        while True:
            # Render the image to the window
            cv2.imshow('Render', img)

            # Wait for a key event
            key = cv2.waitKey(0) & 0xFF
            print(key)
            if key == 27:  # Exit on ESC
                break

            # Update the camera based on the key
            viewpoint_cam = update_camera(key, viewpoint_cam).cuda()

            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            img = render_pkg["render"].detach().clone().permute(1,2,0).cpu().numpy()

    # When everything is done, release the window
    cv2.destroyAllWindows()

    



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--transforms_path", type=str, default = None)
    

    args = parser.parse_args(sys.argv[1:])
        
    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            # Remove the assert statement
            setattr(args, key, host[key])

    for k in cfg.keys():
        recursive_merge(k, cfg)

            
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    viewing(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, 
            args.gaussian_dim, args.time_duration, args.rot_4d, args.force_sh_3d, args)

    # All done
    print("\nDone.")