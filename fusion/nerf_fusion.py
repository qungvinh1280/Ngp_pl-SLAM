#!/usr/bin/env python3

import torch
from lietorch import SE3
import numpy as np
import cv2
from icecream import ic
from utils.flow_viz import *
import os
import sys
import glob
from utils.utils import *
import warnings; warnings.filterwarnings("ignore")

###################################################################
from torch import nn
# from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
import time


from examples.slam_demo import parse_args

# data
from torch.utils.data import DataLoader

from thirdparty.ngp_pl.datasets.nerf import NeRFDataset
from thirdparty.ngp_pl.datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from thirdparty.ngp_pl.models.networks import NGP
from thirdparty.ngp_pl.models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
# from thirdparty.ngp_pl.apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from thirdparty.ngp_pl.losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from thirdparty.ngp_pl.utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")
from thirdparty.ngp_pl.train import NeRFSystem
###################################################################

class NerfFusion:
    def __init__(self, name, args, device) -> None:
        ic(os.environ['CUDA_VISIBLE_DEVICES'])

        self.name = name
        self.args = args
        self.device = device
        self.viz = False
        self.mask_type = args.mask_type # "ours", "ours_w_thresh" or "raw", "no_depth"
        self.split = args.split

    def process_data(self, packet):
        # GROUND_TRUTH Fitting
        self.ngp.nerf.training.optimize_extrinsics = False

        calib = packet["calibs"][0]
        scale, offset = get_scale_and_offset(calib.aabb)
        gt_depth_scale = calib.depth_scale

        packet["poses"]            = scale_offset_poses(np.linalg.inv(packet["poses"]), scale=scale, offset=offset)
        packet["images"]           = (packet["images"].astype(np.float32) / 255.0)
        packet["depths"]           = (packet["depths"].astype(np.float32))
        packet["gt_depths"]        = (packet["depths"].astype(np.float32))
        packet["depth_scale"]      = gt_depth_scale * scale
        packet["depths_cov"]       = np.ones_like(packet["depths"])
        packet["depths_cov_scale"] = 1.0

        self.send_data(packet)
        return False

    def process_slam(self, packet):
        # SLAM_TRUTH Fitting

        # No slam output, just fit for some iters
        if not packet:
            print("Missing fusion input packet from SLAM module...")
            return True

        # Slam output is None, just fit for some iters
        slam_packet = packet[1]
        if slam_packet is None:
            print("Fusion packet from SLAM module is None...")
            return True

        if slam_packet["is_last_frame"]:
            return True

        # Get new data and fit volume
        viz_idx        = slam_packet["viz_idx"]
        cam0_T_world   = slam_packet["cam0_poses"]
        images         = slam_packet["cam0_images"]
        idepths_up     = slam_packet["cam0_idepths_up"]
        depths_cov_up  = slam_packet["cam0_depths_cov_up"]
        calibs         = slam_packet["calibs"]
        gt_depths      = slam_packet["gt_depths"]

        calib = calibs[0]
        scale, offset = get_scale_and_offset(calib.aabb) # if we happen to change aabb, we are screwed...
        gt_depth_scale = calib.depth_scale
        scale = 1.0 # We manually set the scale to 1.0 bcs the automatic get_scale_and_offset sets the scale too small for high-quality recons.
        offset = np.array([0.0, 0.0, 0.0])

        # Mask out depths that have too much uncertainty
        if self.mask_type == "ours":
            pass
        elif self.mask_type == "raw":
            depths_cov_up[...] = 1.0
        elif self.mask_type == "ours_w_thresh":
            masks = (depths_cov_up.sqrt() > depths_cov_up.quantile(0.50))
            idepths_up[masks] = -1.0
        elif self.mask_type == "no_depth":
            idepths_up[...] = -1.0
        else:
            raise NotImplementedError(f"Unknown mask type: {self.mask_type}")

        #TODO: 
        # poses -> matrix
        # images -> [N,H,W,4] float cpu
        # depths -> [N,H,W,1] float cpu up-sampled
        # calibs -> up-sampled
        assert(images.dtype == torch.uint8)
        assert(idepths_up.dtype == torch.float)
        assert(depths_cov_up.dtype == torch.float)

        if self.viz:
            viz_depth_sigma(depths_cov_up.unsqueeze(-1).sqrt(), fix_range=True, bg_img=images, sigma_thresh=20.0, name="Depth Sigma for Fusion")
            cv2.waitKey(1)

        N, _, H, W = images.shape
        alpha_padding = 255 * torch.ones(N, 1, H, W, dtype=images.dtype, device=images.device) # we could avoid this if we didn't remove the alpha channel in the frontend
        images = torch.cat((images, alpha_padding), 1)

        cam0_T_world = SE3(cam0_T_world).matrix().contiguous().cpu().numpy()
        world_T_cam0 = scale_offset_poses(np.linalg.inv(cam0_T_world), scale=scale, offset=offset)
        images = (images.permute(0,2,3,1).float() / 255.0)
        depths = (1.0 / idepths_up[..., None])
        depths_cov = depths_cov_up[..., None]
        gt_depths = gt_depths.permute(0, 2, 3, 1) * gt_depth_scale * scale

        # This is extremely slow.
        # TODO: we could do it in cpp/cuda: send the uint8_t image instead of float, and call srgb_to_linear inside the convert_rgba32 function
        if images.shape[2] == 4:
            images[...,0:3] = srgb_to_linear(images[...,0:3], self.device)
            images[...,0:3] *= images[...,3:4] # Pre-multiply alpha
        else:
            images = srgb_to_linear(images, self.device)

        data_packets = {"k":            viz_idx.cpu().numpy(),
                    "poses":            world_T_cam0,  # needs to be c2w
                    "images":           images.contiguous().cpu().numpy(),
                    "depths":           depths.contiguous().cpu().numpy(),
                    "depth_scale":      scale, # This should be scale, since we scale the poses... # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale,
                    "depths_cov":       depths_cov.contiguous().cpu().numpy(), # do not use up
                    "depths_cov_scale": scale, # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale, 
                    "gt_depths":        gt_depths.contiguous().cpu().numpy(), 
                    "calibs":           calibs,
                }

        self.send_data(data_packets)
        return False

    # Main LOOP
    def fuse(self, data_packets):
        fit = False
        if data_packets:  # data_packets is a dict of data_packets
            for name, packet in data_packets.items():
                if name == "data":
                    fit = self.process_data(packet)
                elif name == "slam":
                    fit = self.process_slam(packet)
                else:
                    raise NotImplementedError(f"process_{name} not implemented...")
            if fit:
                pass
        else:
            pass
        return True  # return None if we want to shutdown

    def send_data(self, batch):
        frame_ids       = batch["k"]
        poses           = batch["poses"]
        images          = batch["images"]
        depths          = batch["depths"]
        depth_scale     = batch["depth_scale"]
        depths_cov      = batch["depths_cov"]
        depth_cov_scale = batch["depths_cov_scale"]
        gt_depths       = batch["gt_depths"]
        calib           = batch["calibs"][0]  # assumes all the same calib

        intrinsics = calib.camera_model.numpy()
        resolution = calib.resolution.numpy()
        focal_length = intrinsics[:2]
        # print('----------------------------------------------------------------------', focal_length)
        principal_point = intrinsics[2:]
        split = self.split
        # print('11111111111111111111111111111111111111111111111111111111111111111111111', principal_point)
        # TODO: we need to restore the self.ref_frames[frame_id] = [image, gt, etc] for evaluation....
        # print('before send', resolution)
        TrainDataset = NeRFDataset(list(frame_ids),
                        list(poses[:, :3, :4]), 
                        list(images), 
                        list(depths), 
                        list(depths_cov), resolution, principal_point, focal_length, depth_scale, depth_cov_scale, split)
        # print(TrainDataset.read_intrinsics)
        # print('-------------------------------------------------------', depths_cov.shape)
    
        # hparams = parse_args()
        # # print(hparams)
        # if hparams.val_only and (not hparams.ckpt_path):
        #     raise ValueError('You need to provide a @ckpt_path for validation!')
        # system = NeRFSystem(hparams)

        # ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
        #                         filename='{epoch:d}',
        #                         save_weights_only=True,
        #                         every_n_epochs=hparams.num_epochs,
        #                         save_on_train_epoch_end=True,
        #                         save_top_k=-1)
        # callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

        # logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
        #                         name=hparams.exp_name,
        #                         default_hp_metric=False)

        # trainer = Trainer(max_epochs=hparams.num_epochs,
        #                 check_val_every_n_epoch=hparams.num_epochs,
        #                 callbacks=callbacks,
        #                 logger=logger,
        #                 enable_model_summary=False,
        #                 accelerator='gpu',
        #                 devices=hparams.num_gpus,
        #                 strategy=DDPPlugin(find_unused_parameters=False)
        #                         if hparams.num_gpus>1 else None,
        #                 num_sanity_val_steps=-1 if hparams.val_only else 0,
        #                 precision=16)

        # trainer.fit(system, ckpt_path=hparams.ckpt_path)

        # if not hparams.val_only: # save slimmed ckpt for the last epoch
        #     ckpt_ = \
        #         slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
        #                 save_poses=hparams.optimize_ext)
        #     torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

        # if not hparams.no_save_test: # save video
        #     imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        #     imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
        #                     [imageio.imread(img) for img in imgs[::2]],
        #                     fps=30, macro_block_size=1)
        #     imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
        #                     [imageio.imread(img) for img in imgs[1::2]],
        #                     fps=30, macro_block_size=1)