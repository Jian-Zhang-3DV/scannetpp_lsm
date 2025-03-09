import argparse
import os
import sys
from pathlib import Path
import imageio
import numpy as np
from tqdm import tqdm
import multiprocessing
from common.utils.colmap import read_model, write_model, Image
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list
from dslr.undistort import compute_undistort_intrinsic
from dslr.downscale import compute_resize_intrinsic
import cv2

def process_scene(scene_id, thread_id, cfg):
    """Process a single scene task running in a separate thread"""
    # Initialize scene and render engine
    try:
        import renderpy
    except ImportError:
        print("renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy")
        sys.exit(1)
    scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
    render_engine = renderpy.Render()
    render_engine.setupMesh(str(scene.scan_mesh_path))

    # Determine render devices
    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # Process each device
    for device in render_devices:
        # Read COLMAP model
        if device == "dslr":
            cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
        else:
            cameras, images, points3D = read_model(scene.iphone_colmap_dir, ".txt")
        assert len(cameras) == 1, "Multiple cameras not supported"
        camera = next(iter(cameras.values()))

        # Get camera parameters
        fx, fy, cx, cy = camera.params[:4]
        params = camera.params[4:]
        camera_model = camera.model
        
        # Undistort
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        height = camera.height
        width = camera.width
        new_K = compute_undistort_intrinsic(K, height, width, params)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, params, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )

        # Adjust resolution
        downscale_factor = float(cfg.get("downscale_factor", 3))
        scale_factor = 1 / downscale_factor
        new_K, new_height, new_width = compute_resize_intrinsic(new_K, height, width, scale_factor)
        new_fx, new_fy, new_cx, new_cy = new_K[0,0], new_K[1,1], new_K[0,2], new_K[1,2]
        new_camera_model = "PINHOLE"
        new_params = np.zeros(4)
        
        # Set render camera
        render_engine.setupCamera(
            new_height, new_width,
            new_fx, new_fy, new_cx, new_cy,
            new_camera_model,
            new_params
        )

        # Set output path
        near = cfg.get("near", 0.05)
        far = cfg.get("far", 20.0)
        depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
        out_image_dir = Path(cfg.output_dir) / scene_id / device / "rgb_resized_undistorted"
        out_mask_dir = Path(cfg.output_dir) / scene_id / device / "mask_resized_undistorted"
        camera_dir = Path(cfg.output_dir) / scene_id / device / "camera"
        
        depth_dir.mkdir(parents=True, exist_ok=True)
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        # Render each image
        for image_id, image in tqdm(images.items(), f"Rendering {device} images for scene {scene_id}"):
            world_to_camera = image.world_to_camera
            _, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
            
            # Save depth map
            depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
            depth_name = image.name.split(".")[0] + ".png"
            imageio.imwrite(depth_dir / depth_name, depth)

            # Save camera parameters
            camera_to_world = np.linalg.inv(world_to_camera)
            camera_params = {
                'intrinsic': new_K.astype(np.float32),
                'extrinsic': camera_to_world.astype(np.float32),
            }
            camera_name = image.name.split(".")[0] + ".npz"
            np.savez(camera_dir / camera_name, **camera_params)

            # Undistort and adjust image and mask size
            image_path = scene.dslr_resized_dir / image.name
            image_rgb = cv2.imread(str(image_path))
            undistorted_image = cv2.remap(image_rgb, map1, map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            resized_image = cv2.resize(
                undistorted_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
            
            mask_path = scene.dslr_resized_mask_dir / depth_name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if np.all(mask > 0):
                undistorted_mask = np.zeros((height, width), dtype=np.uint8) + 255
            else:
                undistorted_mask = cv2.remap(mask, map1, map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )
                undistorted_mask[undistorted_mask < 255] = 0
            resized_mask = cv2.resize(
                undistorted_mask, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
            resized_mask[resized_mask < 255] = 0
            
            # Save processed image and mask
            out_image_path = out_image_dir / image.name
            out_mask_path = out_mask_dir / depth_name
            cv2.imwrite(str(out_image_path), resized_image)
            cv2.imwrite(str(out_mask_path), resized_mask)

def main(args):
    """Main function for task distribution and multi-threaded processing"""
    cfg = load_yaml_munch(args.config_file)

    # Get scene IDs to process
    if cfg.get('scene_list_file'):
        scene_ids = read_txt_list(cfg.scene_list_file)
    elif cfg.get('scene_ids'):
        scene_ids = cfg.scene_ids
    elif cfg.get('splits'):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)

    # Set output directory
    output_dir = cfg.get("output_dir")
    if output_dir is None:
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)
    cfg.output_dir = str(output_dir)  # Ensure output_dir is in cfg

    # Get number of threads to use
    num_threads = cfg.get("num_threads", multiprocessing.cpu_count())

    # Create thread pool and distribute tasks
    with multiprocessing.Pool(processes=num_threads) as pool:
        tasks = [(scene_id, i % num_threads, cfg) for i, scene_id in enumerate(scene_ids)]
        pool.starmap(process_scene, tasks)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    main(args)