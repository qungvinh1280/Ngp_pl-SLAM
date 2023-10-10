#!/usr/bin/env python3

import os
import sys
sys.settrace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.multiprocessing import Process

from datasets.data_module import DataModule
from gui.gui_module import GuiModule
from slam.slam_module import SlamModule
from fusion.fusion_module import FusionModule

from icecream import ic

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Ngp_pl-SLAM")

    # SLAM ARGS
    parser.add_argument("--parallel_run", action="store_true", help="Whether to run in parallel")
    parser.add_argument("--multi_gpu", action="store_true", help="Whether to run with multiple (two) GPUs")
    parser.add_argument("--initial_k", type=int, help="Initial frame to parse in the dataset", default=0)
    parser.add_argument("--final_k", type=int, help="Final frame to parse in the dataset, -1 is all.", default=-1)
    parser.add_argument("--img_stride", type=int, help="Number of frames to skip when parsing the dataset", default=1)
    parser.add_argument("--stereo", action="store_true", help="Use stereo images")
    parser.add_argument("--weights", default="droid.pth", help="Path to the weights file")
    parser.add_argument("--buffer", type=int, default=512, help="Number of keyframes to keep")

    parser.add_argument("--dataset_dir", type=str,
                        help="Path to the dataset directory",
                        default="/home/tonirv/Datasets/euroc/V1_01_easy")
    parser.add_argument('--dataset_name', type=str, default='euroc',
                        choices=['euroc', 'nerf', 'replica', 'real'],
                        help='Dataset format to use.')

    parser.add_argument("--mask_type", type=str, default='ours', choices=['no_depth', 'raw', 'ours', 'ours_w_thresh'])

    #parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument("--slam", action="store_true", help="Run SLAM.")
    parser.add_argument("--fusion", type=str, default='', choices=['tsdf', 'sigma', 'nerf', ''],
                        help="Fusion approach ('' for none):\n\
                            -`tsdf' classical tsdf-fusion using Open3D\n \
                            -`sigma' tsdf-fusion with uncertainty values (Rosinol22wacv)\n \
                            -`nerf' radiance field reconstruction using Instant-NGP.")

    # GUI ARGS
    parser.add_argument("--gui", action="store_true", help="Run O3D Gui, use when volume='tsdf'or'sigma'.")
    parser.add_argument("--width",  "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

    # NERF ARGS
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

    parser.add_argument("--eval", action="store_true", help="Evaluate method.")


    ############################################# Ngp_pl parameter #################################################################
    # dataset parameters
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    
    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')
    
    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    
    #########################################################################################################################

    return parser.parse_args()

def run(args):
    if args.parallel_run and args.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        cpu = 'cpu'
        cuda_slam = 'cuda:0'
        cuda_fusion = 'cuda:1' # you can also try same device as in slam.
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        cpu = 'cpu'
        cuda_slam = cuda_fusion = 'cuda:0'
    print(f"Running with GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not args.parallel_run:
        from queue import Queue
    else:
        from torch.multiprocessing import Queue

    # Create the Queue object

    data_output_queue = Queue()
    slam_output_queue_for_fusion = Queue()


    # Create Dataset provider
    data_provider_module = DataModule(args.dataset_name, args, device=cpu)

    # Create MetaSLAM pipeline
    # The SLAM module takes care of creating the SLAM object itself, to avoid pickling issues
    # (see initialize_module() method inside)
    slam = args.slam
    if slam:
        slam_module = SlamModule("VioSLAM", args, device=cuda_slam)
        data_provider_module.register_output_queue(data_output_queue)
        slam_module.register_input_queue("data", data_output_queue)

    # Create Neural Volume
    fusion = args.fusion != ""
    if fusion:
        fusion_module = FusionModule(args.fusion, args, device=cuda_fusion)
        if slam:
            slam_module.register_output_queue(slam_output_queue_for_fusion)
            fusion_module.register_input_queue("slam", slam_output_queue_for_fusion)
        
    # Run
    if args.parallel_run:
        print("Running pipeline in parallel mode.")

        data_provider_thread = Process(target=data_provider_module.spin, args=())
        if fusion: fusion_thread = Process(target=fusion_module.spin) # FUSION NEEDS TO BE IN A PROCESS


        data_provider_thread.start()
        if fusion: fusion_thread.start()


        # Runs in main thread
        if slam: 
            slam_module.spin() # visualizer should be the main spin, but pytorch has a memory bug/leak if threaded...
            slam_module.shutdown_module()
            ic("Deleting SLAM module to free memory")
            torch.cuda.empty_cache()
            # slam_module.slam. # add function to empty all matrices?
            del slam_module
        print("FINISHED RUNNING SLAM")
        while (fusion and fusion_thread.exitcode == None):
            continue
        print("FINISHED RUNNING FUSION")

        if fusion: fusion_module.shutdown_module()
        data_provider_module.shutdown_module()

        if fusion: fusion_thread.terminate() # violent, should be join()
        data_provider_thread.terminate() # violent, should be a join(), but I don't know how to flush the queue
    else:
        print("Running pipeline in sequential mode.")


        if data_provider_module.spin() \
            and (not slam or slam_module.spin()) \
            and (not fusion or fusion_module.spin()):
            pass

        while data_provider_module.spin() \
            and (not slam or slam_module.spin()) \
            and (not fusion or fusion_module.spin()):
            continue

        ok = True
        while ok:

            if fusion: ok &= fusion_module.spin()



if __name__ == '__main__':
    args = parse_args()

    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    run(args)
    print('Done...')
