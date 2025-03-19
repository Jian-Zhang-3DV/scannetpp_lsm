'''
Generate 2D instance masks from 3D scene data by combining rasterization and semantic projection
'''
from copy import copy
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
from omegaconf import DictConfig
import wandb
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import cv2
import json
import torch.multiprocessing as mp

from pytorch3d.structures import Meshes

from common.utils.colmap import camera_to_intrinsic, get_camera_images_poses
from common.utils.dslr import compute_undistort_intrinsic
from common.utils.rasterize import get_fisheye_cameras_batch, get_opencv_cameras_batch, prep_pt3d_inputs, rasterize_mesh
from common.utils.anno import load_anno_wrapper, get_bboxes_2d
from common.file_io import read_txt_list
from common.utils.image import load_image, viz_ids
from common.scene_release import ScannetppScene_Release

def adjust_intrinsic_matrix(intrinsic, factor):
    # divide fx, fy, cx, cy by factor
    intrinsic /= factor
    intrinsic[2, 2] = 1
    return intrinsic

def process_scene(scene_id, cfg, gpu_id, save_dir_root):
    """Process a single scene on a specific GPU"""
    # Set the GPU device for this process
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)  # Add synchronization
    
    print(f"Processing scene {scene_id} on GPU {gpu_id}")
    
    # Initialize scene
    scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
    
    # Create scene-specific directories
    scene_dir = save_dir_root / scene_id / cfg.image_type
    
    # Create instance mask directories
    instance_mask_dir = scene_dir / 'render_instance'
    instance_viz_dir = scene_dir / 'render_instance_viz' if cfg.visualize_instance_masks else None
    
    instance_mask_dir.mkdir(parents=True, exist_ok=True)
    if instance_viz_dir:
        instance_viz_dir.mkdir(parents=True, exist_ok=True)
        
    print(f'Saving instance masks to: {instance_mask_dir}')
    
    # Read mesh and get vertex object IDs
    mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
    verts, faces, _ = prep_pt3d_inputs(mesh)
    
    # Get object annotations
    anno = load_anno_wrapper(scene)
    vtx_obj_ids = anno['vertex_obj_ids']
    
    # Get unique object IDs
    obj_ids = np.unique(vtx_obj_ids)
    # Remove ID 0 (background)
    obj_ids = sorted(obj_ids[obj_ids != 0])
    print(f'Number of objects in scene: {len(obj_ids)}')
    
    # Convert mesh faces to tensor once per scene - ENSURE DEVICE
    mesh_faces_tensor = torch.tensor(np.array(mesh.triangles), 
                                    dtype=torch.int64, 
                                    device=device)
    
    # Convert vertex object IDs to tensor once per scene - ENSURE DEVICE
    vtx_obj_ids_tensor = torch.tensor(vtx_obj_ids, 
                                      dtype=torch.int64, 
                                      device=device)
    
    # Get camera parameters
    colmap_camera, image_list, poses, distort_params = get_camera_images_poses(scene, cfg.subsample_factor, cfg.image_type)

    # Handle unreleased data prefixes
    if image_list[0].startswith('video/'):
        image_list = [i.split('video/')[-1] for i in image_list]
    if image_list[0].startswith('dslr/'):
        image_list = [i.split('dslr/')[-1] for i in image_list]

    intrinsic_mat = camera_to_intrinsic(colmap_camera)
    img_height, img_width = colmap_camera.height, colmap_camera.width
    
    # Apply undistortion for DSLR fisheye cameras
    if cfg.image_type == 'dslr' and cfg.get('use_undistort', False):
        print('Applying undistortion to DSLR fisheye camera')
        
        # Create undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsic_mat, distort_params[:4], np.eye(3), intrinsic_mat, 
            (img_width, img_height), cv2.CV_32FC1
        )
        
        # Update intrinsic matrix after undistortion
        intrinsic_mat = compute_undistort_intrinsic(intrinsic_mat, img_height, img_width, distort_params[:4])
    
    # Adjust for downsampling if needed
    if cfg.image_downsample_factor:
        img_height = img_height // cfg.image_downsample_factor
        img_width = img_width // cfg.image_downsample_factor
        intrinsic_mat = adjust_intrinsic_matrix(intrinsic_mat, cfg.image_downsample_factor)
    
    # Create batches of images for processing
    batch_size = cfg.batch_size
    batch_start_indices = list(range(0, len(image_list), batch_size))
    
    # Pre-create mesh tensors - ENSURE DEVICE
    with torch.no_grad():
        base_verts = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
        base_faces = torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0)
    
    # Process in batches
    for batch_ndx, batch_start_ndx in enumerate(tqdm(batch_start_indices, desc=f'Scene {scene_id} batches')):
        if cfg.limit_batches and batch_ndx == cfg.limit_batches:
            print(f'Done with {cfg.limit_batches} batches, finish')
            break
        
        # Get batch data - ENSURE DEVICE
        batch_image_list = image_list[batch_start_ndx:batch_start_ndx+batch_size]
        batch_poses = torch.tensor(np.array(poses[batch_start_ndx:batch_start_ndx+batch_size]), 
                                  dtype=torch.float32, 
                                  device=device)  # Explicitly specify device
        
        # Skip empty batches
        if len(batch_image_list) == 0:
            continue
            
        # Skip already processed images if configured
        if cfg.get('skip_existing', False):
            # Create paths for all images in the batch
            mask_paths_batch = [instance_mask_dir / f'{Path(img_name).stem}.png' for img_name in batch_image_list]
            
            # Keep track of original batch
            batch_image_list_orig = copy(batch_image_list)
            batch_poses_orig = batch_poses.clone()
            
            # Filter out existing files
            images_to_process = []
            poses_to_process = []
            
            for i, (img_name, pose, mask_path) in enumerate(zip(batch_image_list, batch_poses, mask_paths_batch)):
                if not mask_path.exists():
                    images_to_process.append(img_name)
                    poses_to_process.append(pose.unsqueeze(0))
            
            # Update batch with only images that need processing
            batch_image_list = images_to_process
            
            if len(poses_to_process) > 0:
                batch_poses = torch.cat(poses_to_process, dim=0)
            else:
                batch_poses = torch.zeros((0,4,4), device=device)
            
            # If nothing left to process, skip this batch
            if len(batch_image_list) == 0:
                print('Skipping batch - all images already processed')
                continue
            
            # Report skipped files
            skipped = set(batch_image_list_orig) - set(batch_image_list)
            if skipped:
                print(f'Skipping {len(skipped)} existing files')
        
        print(f'Images in batch: {batch_image_list}')
        
        # Create camera batch - Make sure all inputs are on the same device
        if cfg.image_type == 'dslr':
            if cfg.get('use_undistort', False):
                # Use undistorted OpenCV cameras for undistorted DSLR
                cameras_batch = get_opencv_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat, device)
            else:
                # Use fisheye cameras for regular DSLR
                # Convert distort_params to tensor on correct device if needed
                if isinstance(distort_params, np.ndarray):
                    distort_params_tensor = torch.tensor(distort_params, device=device)
                else:
                    distort_params_tensor = distort_params
                cameras_batch = get_fisheye_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat, distort_params_tensor)
        elif cfg.image_type == 'iphone':
            cameras_batch = get_opencv_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat)
        
        # Create mesh batch
        bsize = len(batch_image_list)
        if bsize > 1:
            meshes_verts = base_verts.expand(bsize, -1, -1)
            meshes_faces = base_faces.expand(bsize, -1, -1)
        else:
            meshes_verts = base_verts
            meshes_faces = base_faces
        
        # Ensure all mesh components are on the same device
        meshes_verts = meshes_verts.to(device)
        meshes_faces = meshes_faces.to(device)
        
        # Rasterize mesh
        with torch.no_grad():
            meshes_batch = Meshes(verts=meshes_verts, faces=meshes_faces)
        raster_out_dict = rasterize_mesh(meshes_batch, img_height, img_width, cameras_batch, device)
        
        # Process each image in the batch
        for sample_ndx, image_name in enumerate(batch_image_list):
            print(f'Processing image: {image_name}')
            
            # Get pix_to_face for this image and ensure it's on the right device
            pix_to_face = raster_out_dict['pix_to_face'][sample_ndx].squeeze().to(device)
            
            # Adjust face indices
            valid_pix_to_face = pix_to_face[:, :] != -1
            num_faces = meshes_faces.size(1)
            pix_to_face_adjusted = pix_to_face.clone()
            pix_to_face_adjusted[valid_pix_to_face] -= (num_faces * sample_ndx)
            pix_to_face = pix_to_face_adjusted
            
            # Remove file extension from image name for saving output files
            base_image_name = Path(image_name).stem
            
            # Load image
            if cfg.image_type == 'iphone':
                image_dir = scene.iphone_rgb_dir
            elif cfg.image_type == 'dslr':
                image_dir = scene.dslr_resized_dir
            
            img_path = str(image_dir / image_name)
            if not Path(img_path).exists():
                print(f'Image not found: {img_path}, skipping')
                continue
            
            try:
                img = load_image(img_path)
                
                # Apply undistortion if needed
                if cfg.image_type == 'dslr' and cfg.get('use_undistort', False):
                    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                
                # resize image
                img = cv2.resize(img, (img_width, img_height))
            except:
                print(f'Error loading image: {img_path}, skipping')
                continue
            
            # Get object IDs on image
            try:
                # Process on GPU
                pix_to_face_tensor = pix_to_face
                valid_mask = pix_to_face_tensor != -1
                
                # Create output tensor
                pix_obj_ids_tensor = torch.zeros_like(pix_to_face_tensor)
                
                # Assign values for valid pixels
                if valid_mask.any():
                    face_indices = pix_to_face_tensor[valid_mask]
                    vertex_indices = mesh_faces_tensor[face_indices][:, 0]  # Get first vertex of each face
                    pix_obj_ids_tensor[valid_mask] = vtx_obj_ids_tensor[vertex_indices]
                
                # Convert back to CPU for saving
                pix_obj_ids = pix_obj_ids_tensor.cpu().numpy()
            except IndexError:
                print(f'Rasterization error in {scene_id}/{image_name}, skipping')
                continue
            
            # Create instance mask
            instance_mask = pix_obj_ids.astype(np.uint16)  # Use uint16 to support more instance IDs
            
            # Save instance mask with cleaned filename
            mask_path = instance_mask_dir / f'{base_image_name}.png'
            print(f'Saving instance mask to {mask_path}')
            cv2.imwrite(str(mask_path), instance_mask)
            
            # Create visualization with cleaned filename
            if cfg.visualize_instance_masks:
                viz_path = instance_viz_dir / f'{base_image_name}.png'
                print(f'Saving visualization to {viz_path}')
                viz_ids(img, pix_obj_ids, viz_path)
            
            # Get bounding boxes if needed, with cleaned filename
            if cfg.save_instance_bboxes:
                bboxes_2d = get_bboxes_2d(pix_obj_ids)
                bbox_path = instance_mask_dir / f'{base_image_name}_bboxes.json'
                with open(bbox_path, 'w') as f:
                    json.dump(bboxes_2d, f)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    print(f"Scene {scene_id} processing complete on GPU {gpu_id}")
    return scene_id

@hydra.main(version_base=None, config_path="../configs", config_name="instance_mask_2d")
def main(cfg : DictConfig) -> None:
    print('Config:', cfg)
    
    if not cfg.no_log:
        wandb.init(project='instance_mask_generation', 
                   group=cfg.wandb_group, config=OmegaConf.to_container(cfg, resolve=True), notes=cfg.wandb_notes)

    # get scene list
    scene_list = read_txt_list(cfg.scene_list_file)
    print('Scenes in list:', len(scene_list))

    if cfg.get('filter_scenes'):
        scene_list = [s for s in scene_list if s in cfg.filter_scenes]
        print('Filtered scenes:', len(scene_list))
    if cfg.get('exclude_scenes'):
        scene_list = [s for s in scene_list if s not in cfg.exclude_scenes]
        print('After excluding scenes:', len(scene_list))

    # Create output directories
    save_dir_root = Path(cfg.save_dir_root)
    print('Save root dir:', save_dir_root)
    
    # Temporary directory for rasterization results if needed
    if cfg.save_raster_results:
        rasterout_dir = Path(cfg.rasterout_dir) / cfg.image_type
        rasterout_dir.mkdir(parents=True, exist_ok=True)
    else:
        rasterout_dir = None

    # Determine GPUs to use
    if cfg.get('gpu_ids'):
        gpu_ids = cfg.gpu_ids
        print(f"Using GPU IDs specified in config: {gpu_ids}")
    else:
        # Default: use all available GPUs
        num_gpus = torch.cuda.device_count()
        if cfg.get('num_gpus'):
            num_gpus = min(num_gpus, cfg.num_gpus)
        gpu_ids = list(range(num_gpus))
        print(f"Using {num_gpus} GPUs for processing: {gpu_ids}")
    
    if len(gpu_ids) > 1:
        # Multi-GPU processing
        mp.set_start_method('spawn', force=True)
        
        # Create a pool of workers, one per GPU
        pool = mp.Pool(len(gpu_ids))
        
        # Assign scenes to GPUs in a round-robin fashion
        results = []
        for i, scene_id in enumerate(scene_list):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            results.append(pool.apply_async(process_scene, args=(scene_id, cfg, gpu_id, save_dir_root)))
        
        # Close the pool and wait for all processes to finish
        pool.close()
        
        # Monitor progress
        finished_scenes = set()
        with tqdm(total=len(scene_list), desc="Total progress") as pbar:
            while len(finished_scenes) < len(scene_list):
                for i, result in enumerate(results):
                    if result.ready() and i not in finished_scenes:
                        scene_id = result.get()
                        finished_scenes.add(i)
                        pbar.update(1)
                        print(f"Completed scene {scene_id}")
                
                if len(finished_scenes) < len(scene_list):
                    import time
                    time.sleep(5)  # Check again after a short delay
        
        pool.join()
        
    else:
        # Single GPU processing
        gpu_id = gpu_ids[0] if gpu_ids else 0
        print(f"Using single GPU: {gpu_id}")
        for scene_id in tqdm(scene_list, desc='scene'):
            process_scene(scene_id, cfg, gpu_id, save_dir_root)
    
    print('Instance mask generation complete!')

if __name__ == "__main__":
    main() 