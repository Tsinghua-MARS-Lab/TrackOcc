import os
from mayavi import mlab
import mayavi
import imageio
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
import argparse, torch, os, json
import shutil
import numpy as np
import torch.nn.functional as F
import pickle as pkl
import argparse


panoptic_colormap = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [200, 100,   0, 255], #[255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bicycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [128, 128, 128, 255], # 16 for vis  **
        [255, 255, 0, 255], # 17 yellow
        [0, 255, 0, 255], # 18 lime
        [0, 255, 255, 255], # 19 cyan
        [0, 0, 255, 255], # 20 blue  ** 
        [255, 0, 255, 255], # 21 magenta
        [128, 0, 128, 255], # 22 purple
        [128, 0, 0, 255], # 23 maroon
        [255, 255, 255, 255], # 24 white
        [192, 192, 192, 255], # 25 silver
        [128, 128, 0, 255], # 26 olive    **
        [0, 128, 128, 255], # 27 teal
        [128, 128, 255, 255], # 28 light blue
        [255, 128, 0, 255], # 29 orange
        [128, 255, 0, 255], # 30 light green
        [255, 0, 128, 255], # 31 pink
        [0, 128, 0, 255], # 32 green
        [0, 0, 128, 255], # 33 navy
        [128, 0, 255, 255], # 34 violet
        [255, 128, 128, 255], # 35 light red    **
        [128, 255, 128, 255], # 36 light lime
        [128, 255, 255, 255], # 37 light cyan
        [128, 128, 255, 255], # 38 medium blue  **
        [255, 128, 255, 255], # 39 light magenta
        [192, 0, 192, 255], # 40 fuchsia
        [192, 192, 0, 255], # 41 yellow green
        [192, 0, 0, 255], # 42 dark red
        [0, 192, 0, 255], # 43 dark green
        [0, 192, 192, 255], # 44 dark cyan
        [0, 0, 192, 255], # 45 dark blue   **
        [192, 0, 128, 255], # 46 dark magenta
        [0, 128, 192, 255], # 47 medium cyan
        [128, 0, 192, 255], # 48 medium magenta
        [192, 128, 0, 255], # 49 dark yellow
        [192, 192, 128, 255], # 50 medium yellow
        [128, 192, 0, 255], # 51 medium olive
        [0, 192, 128, 255], # 52 medium teal
        [0, 128, 192, 255], # 53 medium navy
        [128, 0, 64, 255], # 54 dark maroon
        [128, 0, 32, 255], # 55 dark maroon (darker)
        [128, 64, 0, 255], # 56 olive brown
        [64, 128, 0, 255], # 57 dark lime
        [0, 128, 64, 255], # 58 dark teal
        [0, 64, 128, 255], # 59 dark navy
        [64, 0, 128, 255], # 60 dark purple
        [128, 0, 96, 255], # 61 dark pink
        [128, 64, 32, 255], # 62 brown
        [64, 128, 32, 255], # 63 medium lime
        [32, 128, 64, 255], # 64 medium teal
        [32, 64, 128, 255], # 65 medium navy
        [64, 32, 128, 255], # 66 medium purple
        [128, 32, 96, 255], # 67 medium pink
        [192, 64, 32, 255], # 68 light brown
        [64, 192, 32, 255], # 69 light lime
        [32, 192, 64, 255], # 70 light teal
        [32, 64, 192, 255], # 71 light navy
        [64, 32, 192, 255], # 72 light
        [128, 64, 192, 255], # 73 light purple
        [192, 32, 128, 255], # 74 light pink
        [192, 96, 32, 255], # 75 orange brown
        [32, 192, 96, 255], # 76 light lime
        [96, 32, 192, 255], # 77 light purple
        [192, 32, 96, 255], # 78 dark pink
        [96, 192, 32, 255], # 79 light lime
        [32, 96, 192, 255], # 80 light purple
        [96, 32, 192, 255], # 81 dark pink
        [255, 0, 255, 255], # 82 magenta
        [0, 255, 255, 255], # 83 cyan
        [255, 255, 0, 255], # 84 yellow
        [255, 128, 0, 255], # 85 orange
        [255, 0, 128, 255], # 86 pink
        [128, 255, 0, 255], # 87 lime
        [0, 255, 128, 255], # 88 light green
        [128, 0, 255, 255], # 89 purple
        [0, 128, 255, 255], # 90 light blue
        [255, 128, 128, 255], # 91 light red
        [255, 158, 0, 255], # orange #[128, 255, 128, 255], # 92 light lime
        [128, 128, 255, 255], # 93 light blue
        [255, 128, 255, 255], # 94 light magenta
        [255, 0, 128, 255], # 95 dark pink
        [128, 0, 255, 255], # 96 dark purple
        [0, 128, 255, 255], # 97 dark blue
        [255, 255, 128, 255], # 98 light yellow
        [128, 255, 255, 255], # 99 light cyan
        [255, 128, 255, 255], # 100 light magenta
        [192, 0, 255, 255], # 101 violet
        [255, 192, 0, 255], # 102 gold
        [255, 0, 192, 255], # 103 fuchsia
        [0, 255, 192, 255], # 104 light teal
        [192, 255, 0, 255], # 105 light lime
        [192, 0, 255, 255], # 106 dark purple
        [255, 192, 192, 255], # 107 light pink
        [192, 255, 192, 255], # 108 light lime
        [192, 192, 255, 255], # 109 light periwinkle
        [255, 192, 192, 255], # 110 soft red
        [192, 255, 255, 255], # 111 pale cyan
        [255, 255, 192, 255], # 112 pale yellow
        [255, 224, 192, 255], # 113 peach
        [192, 255, 224, 255], # 114 mint
        [224, 192, 255, 255], # 115 soft purple
        [255, 192, 224, 255], # 116 soft pink
        [224, 255, 192, 255], # 117 pale lime
        [192, 224, 255, 255], # 118 soft blue
        [224, 224, 192, 255], # 119 pale olive
        [192, 224, 224, 255], # 120 pale aqua
        [224, 192, 224, 255], # 121 pale lavender
        [224, 224, 224, 255], # 122 light grey
        [233, 233, 233, 255], # 123 very light grey
        [233, 200, 200, 255], # 124 soft rose
        [200, 233, 200, 255], # 125 soft mint
        [200, 200, 233, 255], # 126 soft sky blue
        [233, 200, 233, 255], # 127 soft lilac
        [200, 233, 233, 255], # 128 soft turquoise
        [233, 230, 140, 255], # 129 soft khaki
        [230, 230, 250, 255], # 130 lavender blush
        [250, 235, 215, 255], # 131 antique white
        [0, 255, 255, 255],   # 132 cyan
        [127, 255, 212, 255], # 133 aquamarine
        [240, 128, 128, 255], # 134 light coral
        [255, 99, 71, 255],   # 135 tomato
        [64, 224, 208, 255],  # 136 turquoise
        [218, 165, 32, 255],  # 137 golden rod
        [210, 105, 30, 255],  # 138 chocolate
        [220, 20, 60, 255],   # 139 crimson
        [255, 105, 180, 255], # 140 hot pink
        [147, 112, 219, 255], # 141 medium purple
        [60, 179, 113, 255],  # 142 medium sea green
        [32, 178, 170, 255],  # 143 light sea green
        [95, 158, 160, 255],  # 144 cadet blue
        [70, 130, 180, 255],  # 145 steel blue
        [100, 149, 237, 255], # 146 cornflower blue
        [123, 104, 238, 255], # 147 medium slate blue
        [0, 191, 255, 255],   # 148 deep sky blue
        [30, 144, 255, 255],  # 149 dodger blue
        [135, 206, 235, 255] # 150 sky blue
    ]
)

# Control the result's perspective
VIS_NORMAL = True
VIS_CAM = False
VIS_BEV = False

# Always true
VALIDATION = True
# if True, visualize the ground truth
VIS_GT = True

save_camera_params = False

use_coarse_voxel = True

# for voxel04
if not use_coarse_voxel:
    VOXEL_SIZE=[0.05, 0.05, 0.05]
    SPTIAL_SHAPE=[1600, 1600, 128]
    FREE_LABEL = 23
else:
    VOXEL_SIZE=[0.4, 0.4, 0.4]
    SPTIAL_SHAPE=[200, 200, 16]
    FREE_LABEL = 15 #23

def gridcloud3d(B, Z, Y, X, device='cpu'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    # pdb.set_trace()
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # here is stack in order with xyz
    # this is B x N x 3

    # pdb.set_trace()
    return xyz


def meshgrid3d(B, Z, Y, X, stack=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)
    # here repeat is in the order with ZYX

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def draw(
    voxels,          # semantic occupancy predictions
    vox_origin,      #
    voxel_size=0.4,  # voxel size in the real world
    pc_range=[-40, -40, -1, 40, 40, 5.4],        # point cloud range
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    voxels_lidar=None,
    save_dirs=None,
    str_frame_id=None,
    cam_positions=None,
    focal_positions=None,
    cam_names=None,
    timestamp=None,
    mode=0,          # mode:0 pred, 1 gt
):
    os.makedirs(save_dirs, exist_ok=True)

    h, w, z = voxels.shape
    if grid is not None:
        grid = grid.astype(np.int)

    X_org, Y_org, Z_org = voxels.shape
    x_offset, y_offset, z_offset = (X_org - 200) // 2, (Y_org - 200) // 2, (Z_org - 16) // 2
    voxels = voxels[x_offset:x_offset+200, y_offset:y_offset+200, z_offset:z_offset+16]
    voxels_lidar = voxels_lidar[x_offset:x_offset+200, y_offset:y_offset+200, z_offset:z_offset+16]

    xyz = gridcloud3d(1, 16, 200, 200, device='cpu')
    xyz_min = np.array(pc_range[:3])
    xyz_max = np.array(pc_range[3:])
    occ_size = np.array([200, 200, 16])
    xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * voxel_size
    xyz = xyz.reshape(16, 200, 200, 3).permute(2, 1, 0, 3).numpy()
    xyz_lidar = np.concatenate([xyz, voxels_lidar[..., None]], axis=-1).reshape(-1, 4)
    xyz = np.concatenate([xyz, voxels[..., None]], axis=-1).reshape(-1, 4)

    # if mode == 0: # occupancy pred
    # grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    grid_coords = xyz
    # grid_coords[grid_coords[:, 3] == FREE_LABEL, 3] = 20
    grid_coords_lidar = xyz_lidar
    # grid_coords_lidar[grid_coords_lidar[:, 3] == FREE_LABEL, 3] = 20

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords
    fov_grid_coords_lidar = grid_coords_lidar

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] != FREE_LABEL)
    ]
    fov_voxels_lidar = fov_grid_coords_lidar[
        (fov_grid_coords_lidar[:, 3] >= 0) & (fov_grid_coords_lidar[:, 3] != FREE_LABEL)
    ]
    
    colors = panoptic_colormap.astype(np.uint8)

    if VIS_CAM:
        figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
        # Draw occupied inside FOV voxels
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=0.95 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=150, # 19
        )
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        scene = figure.scene
        
        for i, cam_name in enumerate(cam_names):
            # if cam_name != 'CAM_FRONT_LEFT':
            #     continue
            scene.camera.position = cam_positions[i]
            scene.camera.focal_point = focal_positions[i] 
            scene.camera.view_angle = 35 if i != 3 else 60
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()
            if mode == 0:
                save_path = os.path.join(save_dirs, f'{str_frame_id}_{cam_name}.png')
            elif mode == 1:
                save_path = os.path.join(save_dirs, f'{str_frame_id}_{cam_name}.png')
            # print(f"save_path:{save_path}")
            mlab.savefig(save_path)
        mlab.close(scene=None, all=False)
    
    if VIS_NORMAL:
        figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
        
        plt_plot_fov_lidar = mlab.points3d(
            fov_voxels_lidar[:, 0],
            fov_voxels_lidar[:, 1],
            fov_voxels_lidar[:, 2],
            fov_voxels_lidar[:, 3],
            colormap="viridis",
            scale_factor=0.95 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=150, # 19
        )
        plt_plot_fov_lidar.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov_lidar.module_manager.scalar_lut_manager.lut.table = colors
        scene = figure.scene
        
        scene.camera.position = [-1.6323111397440937,
            -21.353753494039857,
            44.52780994024775] #[0.75131739, -35.08337438,  16.71378558]
        scene.camera.focal_point = [-1.7412819315994255,
            -9.528452124286424,
            30.806294872650476] #[0.75131739, -34.21734897,  16.21378558]
        scene.camera.view_angle = 40.0
        scene.camera.view_up = [ -0.005634634398811905,
            0.7574738369834945,
            0.6528412036480955]
        scene.camera.clipping_range = [ 10.938169311856104,
            107.41292358309394]
        scene.camera.compute_view_plane_normal()
        scene.render()

        if save_camera_params:
            mlab.show()
            # 获取并保存摄像机参数
            camera_params = {
            'position': list(scene.camera.position),      # 转换为列表
            'focal_point': list(scene.camera.focal_point),  # 转换为列表
            'view_angle': scene.camera.view_angle,
            'view_up': list(scene.camera.view_up),        # 转换为列表
            'clipping_range': list(scene.camera.clipping_range)  # 转换为列表
        }
            # 保存到文件
            with open('camera_params.json', 'w') as f:
                json.dump(camera_params, f, indent=4)
        
        if mode==0:
            save_path = os.path.join(save_dirs, f'{str_frame_id}_normal.png')
        elif mode==1:
            save_path = os.path.join(save_dirs, f'{str_frame_id}_normal.png')
        # print(f"save_path:{save_path}")
        mlab.savefig(save_path)
        mlab.close(scene=None, all=False)

    if VIS_BEV:
        figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
        plt_plot_fov_lidar = mlab.points3d(
            fov_voxels_lidar[:, 0],
            fov_voxels_lidar[:, 1],
            fov_voxels_lidar[:, 2],
            fov_voxels_lidar[:, 3],
            colormap="viridis",
            scale_factor=0.95 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=150, # 19
        )
        plt_plot_fov_lidar.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov_lidar.module_manager.scalar_lut_manager.lut.table = colors
        scene = figure.scene

        scene.camera.position = [ 0.75131739,  10, 93.21378558] #[ 0.75131739,  0.78265103, 93.21378558]
        scene.camera.focal_point = [ 0.75131739,  10, 92.21378558] #[ 0.75131739,  0.78265103, 92.21378558]
        scene.camera.view_angle = 40.0
        scene.camera.view_up = [0., 1., 0.]
        scene.camera.clipping_range = [0.01, 400.]
        scene.camera.compute_view_plane_normal()
        scene.render()
        if mode==0:
            save_path = os.path.join(save_dirs, f'{str_frame_id}_bev.png')
        elif mode==1:
            save_path = os.path.join(save_dirs, f'{str_frame_id}_bev.png')
        # print(f"save_path:{save_path}")
        mlab.savefig(save_path)
        mlab.close(scene=None, all=False)

def make_gif(image_dir, save_path, fps=5):
    images = []
    files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(".png")
    ])

    for f in files:
        img_path = os.path.join(image_dir, f)
        images.append(imageio.imread(img_path))

    if len(images) > 0:
        imageio.mimsave(save_path, images, fps=fps)
        print(f"GIF saved to: {save_path}")
    else:
        print(f"No images found in {image_dir}")

def make_side_by_side_gif(pred_dir, gt_dir, save_path, fps=5, resize=None):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    gt_files   = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])

    images = []

    for pf, gf in zip(pred_files, gt_files):
        pred_img = imageio.imread(os.path.join(pred_dir, pf))
        gt_img   = imageio.imread(os.path.join(gt_dir, gf))

        # 可选 resize（避免 GIF 太大）
        if resize is not None:
            import cv2
            pred_img = cv2.resize(pred_img, resize)
            gt_img   = cv2.resize(gt_img, resize)

        # 确保是 RGB
        if pred_img.shape[-1] == 4:
            pred_img = pred_img[..., :3]
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[..., :3]

        # 左右拼接
        concat = np.concatenate([pred_img, gt_img], axis=1)
        images.append(concat)

    if len(images) > 0:
        imageio.mimsave(save_path, images, fps=fps)
        print(f"Saved side-by-side GIF: {save_path}")
    else:
        print("No frames found.")

def vis_04(vis_save_path, waymo_pkl, gt_occ_dir, pred_dir, scene_idx=90):
    
    # load pickle file
    waymo_data = pkl.load(open(waymo_pkl, 'rb'))
    data_infos = waymo_data['data_list']
    # formute dict: sample_idx to index
    sampled_idx2index = {}
    for idx, info in enumerate(data_infos):
        sampled_idx2index[info['sample_idx']] = idx
    
    for idx in range(0, 185, 5):
        occ_file = os.path.join(gt_occ_dir, str(scene_idx).zfill(3), f'{str(idx).zfill(3)}_04.npz')
        if VALIDATION:
            sampled_idx_str = f'1{str(scene_idx).zfill(3)}{str(idx).zfill(3)}'
            
        else:
            sampled_idx_str = f'0{str(scene_idx).zfill(3)}{str(idx).zfill(3)}'
        sampled_idx = int(sampled_idx_str)
        info = data_infos[sampled_idx2index[sampled_idx]]

        if VALIDATION:
            scene_dir = os.path.join(vis_save_path, f'1{str(scene_idx).zfill(3)}')
        else:
            scene_dir = os.path.join(vis_save_path, f'0{str(scene_idx).zfill(3)}')
        os.makedirs(scene_dir, exist_ok=True)

        occ_npz = np.load(occ_file)
        gt_sem_vox = occ_npz['semantics'] # [200, 200, 16] array
        gt_ins_vox = occ_npz['instances'] # value=0 means empty. from 1, not continuous
        visible_mask = occ_npz['mask_camera'].astype(bool)
        infov = occ_npz['infov'].astype(bool)
        visible_mask = visible_mask & infov
        ego2global = occ_npz['ego2global']
        

        # for cls in [1, 2, 4]: # foreground do not use visible mask
        #     mask = (gt_sem_vox == cls) & infov
        #     visible_mask[mask] = True
        
        gt_sem_vox[~visible_mask] = FREE_LABEL

        voxel_ins_sem_label_gt = np.zeros_like(gt_sem_vox)
        inst_mask = np.logical_and(gt_ins_vox != 0, gt_ins_vox != 3)
        voxel_ins_sem_label_gt[inst_mask] = gt_ins_vox[inst_mask] + 16
        stuff_mask = np.logical_and(gt_sem_vox != 1, gt_sem_vox != 2)
        stuff_mask = np.logical_and(stuff_mask, gt_sem_vox != 4)
        voxel_ins_sem_label_gt[stuff_mask] = gt_sem_vox[stuff_mask]

        
        # pred
        pred_vox_dir = os.path.join(pred_dir, f'{sampled_idx}.npz')
        pred_npz = np.load(pred_vox_dir)
        pred_sem_vox = pred_npz['pano_sem']
        pred_ins_vox = pred_npz['pano_inst']
        pred_sem_vox[~visible_mask] = FREE_LABEL

        voxel_ins_sem_label_pred = np.zeros_like(pred_sem_vox)
        inst_mask = np.logical_and(pred_ins_vox != 0, pred_ins_vox != 3)
        voxel_ins_sem_label_pred[inst_mask] = pred_ins_vox[inst_mask] + 16
        stuff_mask = np.logical_and(pred_sem_vox != 1, pred_sem_vox != 2)
        stuff_mask = np.logical_and(stuff_mask, pred_sem_vox != 4)
        voxel_ins_sem_label_pred[stuff_mask] = pred_sem_vox[stuff_mask]

        cam_positions = []
        focal_positions = []
        cam_names = []

        for cam_type, cam_info in info['images'].items():
            cam_names.append(cam_type)
            # get cam2ego from ego2cam
            cam2ego = np.linalg.inv(cam_info['lidar2cam'])

            f = 0.0055
            cam_position = cam2ego @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = cam2ego @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        voxel_origin = [-40, -40, -1.0]
        voxel_max = [40.0, 40.0, 5.4]
        grid_size = [200, 200, 16]
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

        gt_vox_lidar = voxel_ins_sem_label_gt.copy().transpose(2, 0, 1)  # h w z -> z h w
        gt_vox_lidar = np.rot90(gt_vox_lidar, 1, [1, 2])
        gt_vox_lidar = gt_vox_lidar.transpose(1, 2, 0)

        pred_vox_lidar = voxel_ins_sem_label_pred.copy().transpose(2, 0, 1)  # h w z -> z h w
        pred_vox_lidar = np.rot90(pred_vox_lidar, 1, [1, 2])
        pred_vox_lidar = pred_vox_lidar.transpose(1, 2, 0)

        draw(voxel_ins_sem_label_pred,
             voxel_origin,
             voxels_lidar=pred_vox_lidar,
             save_dirs=os.path.join(scene_dir, 'pred'),
             str_frame_id=sampled_idx,
             cam_positions=cam_positions,
             focal_positions=focal_positions,
             cam_names=cam_names,
             mode=0)
        if VIS_GT:
            draw(voxel_ins_sem_label_gt,
                voxel_origin,
                voxels_lidar=gt_vox_lidar,
                save_dirs=os.path.join(scene_dir, 'gt'),
                str_frame_id=sampled_idx,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                cam_names=cam_names,
                mode=1)

    # format gif
    if VALIDATION:
        scene_dir = os.path.join(vis_save_path, f'1{str(scene_idx).zfill(3)}')
    else:
        scene_dir = os.path.join(vis_save_path, f'0{str(scene_idx).zfill(3)}')

    pred_dir_path = os.path.join(scene_dir, 'pred')
    gt_dir_path = os.path.join(scene_dir, 'gt')

    # 生成 GIF
    make_gif(
        pred_dir_path,
        os.path.join(scene_dir, f"{scene_idx}_pred.gif"),
        fps=5
    )

    if VIS_GT:
        make_gif(
            gt_dir_path,
            os.path.join(scene_dir, f"{scene_idx}_gt.gif"),
            fps=5
        )
        
        make_side_by_side_gif(
            pred_dir_path,
            gt_dir_path,
            os.path.join(scene_dir, f"{scene_idx}_pred_gt.gif"),
            fps=5,
            resize=(960, 540)   # 可选：防止太大
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the results of TrackOcc')
    parser.add_argument('--vis_save_path', default="./test_results/vis", help='vis save path')
    parser.add_argument('--waymo_pkl', default="./data/TrackOcc-waymo/kitti_format/waymo_infos_val_jpg.pkl", help='waymo pkl file')
    parser.add_argument('--gt_occ_dir', default="./data/TrackOcc-waymo/pano_voxel04/validation", help='gt occ dir')
    parser.add_argument('--pred_dir', default="./test_results/occupancy_pred", help='predition result dir')
    parser.add_argument('--scene_idx', default=90, help='scene idx')
    
    args = parser.parse_args()
    vis_04(
        vis_save_path=args.vis_save_path,
        waymo_pkl=args.waymo_pkl,
        gt_occ_dir=args.gt_occ_dir,
        pred_dir=args.pred_dir,
        scene_idx=args.scene_idx,
    )