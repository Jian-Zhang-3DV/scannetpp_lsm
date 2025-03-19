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

from pytorch3d.structures import Meshes

from common.utils.colmap import camera_to_intrinsic, get_camera_images_poses
from common.utils.dslr import compute_undistort_intrinsic
from common.utils.rasterize import get_fisheye_cameras_batch, get_opencv_cameras_batch, prep_pt3d_inputs, rasterize_mesh
from common.utils.anno import get_vtx_prop_on_2d, load_anno_wrapper, get_bboxes_2d
from common.file_io import read_txt_list
from common.utils.image import load_image, save_img, viz_ids
from common.scene_release import ScannetppScene_Release

def adjust_intrinsic_matrix(intrinsic, factor):
    # divide fx, fy, cx, cy by factor
    intrinsic /= factor
    intrinsic[2, 2] = 1
    return intrinsic

device = torch.device("cuda:0")

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

    # Process each scene
    for scene_id in tqdm(scene_list, desc='scene'):
        print(f'Processing scene: {scene_id}')
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
        
        # Convert mesh faces to tensor once per scene
        mesh_faces_tensor = torch.tensor(np.array(mesh.triangles), 
                                        dtype=torch.int64, 
                                        device=device)
        
        # Convert vertex object IDs to tensor once per scene
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
        
        # Pre-create mesh tensors
        with torch.no_grad():
            base_verts = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
            base_faces = torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0)
        
        # Process in batches
        for batch_ndx, batch_start_ndx in enumerate(tqdm(batch_start_indices, desc='batch')):
            if cfg.limit_batches and batch_ndx == cfg.limit_batches:
                print(f'Done with {cfg.limit_batches} batches, finish')
                break
            
            # Get batch data
            batch_image_list = image_list[batch_start_ndx:batch_start_ndx+batch_size]
            batch_poses = torch.Tensor(np.array(poses[batch_start_ndx:batch_start_ndx+batch_size]))
            
            # Skip empty batches
            if len(batch_image_list) == 0:
                continue
                
            print(f'Images in batch: {batch_image_list}')
            
            # Create camera batch
            if cfg.image_type == 'dslr':
                if cfg.get('use_undistort', False):
                    # Use undistorted OpenCV cameras for undistorted DSLR
                    cameras_batch = get_opencv_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat)
                else:
                    # Use fisheye cameras for regular DSLR
                    cameras_batch = get_fisheye_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat, distort_params)
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
            
            # Rasterize mesh
            with torch.no_grad():
                meshes_batch = Meshes(verts=meshes_verts, faces=meshes_faces)
            raster_out_dict = rasterize_mesh(meshes_batch, img_height, img_width, cameras_batch)
            
            # Process each image in the batch
            for sample_ndx, image_name in enumerate(batch_image_list):
                print(f'Processing image: {image_name}')
                
                # Get pix_to_face for this image
                pix_to_face = raster_out_dict['pix_to_face'][sample_ndx].squeeze()
                
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
    
    print('Instance mask generation complete!')

if __name__ == "__main__":
    main() 