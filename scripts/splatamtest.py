import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# SUBMAP: импортируем класс Submap
from utils.submap import Submap

# ---------------------------------------------------------------------------- #
#                            ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                           #
# ---------------------------------------------------------------------------- #

def get_dataset(config_dict, basedir, sequence, **kwargs):
    # ... (без изменений, оставляем как есть)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    # ... (без изменений)
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
    point_cld = torch.cat((pts, cols), -1)

    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    # Эту функцию мы пока оставляем, но она больше не будет использоваться для создания глобальных params.
    # Она может пригодиться для инициализации новой подкарты. В классе Submap мы реализовали аналогичную логику,
    # но если хотим использовать эту функцию, нужно будет передавать num_frames? В подкарте num_frames не нужен,
    # так как камерные параметры хранятся отдельно. Оставим её для совместимости, но в новом коде не используем.
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Здесь убираем инициализацию камерных параметров, так как они будут глобальными.
    # Раньше было:
    # cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    # cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    # params['cam_unnorm_rots'] = cam_rots
    # params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


# SUBMAP: новая функция для инициализации позы камеры с глобальными тензорами
def initialize_camera_pose_global(cam_rots, cam_trans, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Инициализация по модели постоянной скорости
            prev_rot1 = F.normalize(cam_rots[..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(cam_rots[..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            cam_rots[..., curr_time_idx] = new_rot.detach()
            prev_tran1 = cam_trans[..., curr_time_idx-1].detach()
            prev_tran2 = cam_trans[..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            cam_trans[..., curr_time_idx] = new_tran.detach()
        else:
            cam_rots[..., curr_time_idx] = cam_rots[..., curr_time_idx-1].detach()
            cam_trans[..., curr_time_idx] = cam_trans[..., curr_time_idx-1].detach()
    return cam_rots, cam_trans


# SUBMAP: модифицированная версия get_loss, работающая с отдельными тензорами камеры
def get_loss_with_cam(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
                      sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
                      mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False,
                      tracking_iteration=None, cam_rots=None, cam_trans=None):
    """
    Аналог оригинальной get_loss, но камерные параметры передаются отдельно (cam_rots, cam_trans).
    Внутри используем их вместо params['cam_...'].
    """
    losses = {}

    # Определяем, какие градиенты нужны
    if tracking:
        gaussians_grad = False
        camera_grad = True
    elif mapping:
        if do_ba:
            gaussians_grad = True
            camera_grad = True
        else:
            gaussians_grad = True
            camera_grad = False
    else:
        gaussians_grad = True
        camera_grad = False

    # Трансформируем гауссианы с использованием переданных параметров камеры
    # Используем transform_to_frame, но нужно передать cam_rots и cam_trans.
    # Для этого либо модифицируем transform_to_frame, либо используем локальную версию.
    # Пока создадим локальную функцию transform_to_frame_local, которая принимает cam_rots, cam_trans.
    def transform_to_frame_local(params, time_idx, gaussians_grad, camera_grad, cam_rots, cam_trans):
        if camera_grad:
            cam_rot = F.normalize(cam_rots[..., time_idx])
            cam_tran = cam_trans[..., time_idx]
        else:
            cam_rot = F.normalize(cam_rots[..., time_idx].detach())
            cam_tran = cam_trans[..., time_idx].detach()
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran

        if params['log_scales'].shape[1] == 1:
            transform_rots = False
        else:
            transform_rots = True

        if gaussians_grad:
            pts = params['means3D']
            unnorm_rots = params['unnorm_rotations']
        else:
            pts = params['means3D'].detach()
            unnorm_rots = params['unnorm_rotations'].detach()

        transformed_gaussians = {}
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
        transformed_gaussians['means3D'] = transformed_pts

        if transform_rots:
            # функция quat_mult должна быть доступна (импортирована из slam_helpers)
            from utils.slam_helpers import quat_mult
            norm_rots = F.normalize(unnorm_rots)
            transformed_rots = quat_mult(cam_rot, norm_rots)
            transformed_gaussians['unnorm_rotations'] = transformed_rots
        else:
            transformed_gaussians['unnorm_rotations'] = unnorm_rots

        return transformed_gaussians

    transformed_gaussians = transform_to_frame_local(params, iter_time_idx, gaussians_grad, camera_grad, cam_rots, cam_trans)

    # Далее код практически идентичен оригинальной get_loss, но используем переданные cam_rots и cam_trans там, где нужно.
    # В оригинале внутри get_loss были обращения к params['cam_...'] только в визуализации и при построении curr_w2c.
    # В визуализации они используются для отладки, но мы пока упростим: уберем зависимость от них, либо заменим на переданные.
    # Для простоты пока не будем использовать визуализацию, если tracking и visualize_tracking_loss, то пропустим эту часть.
    # В остальном код одинаков.

    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'], transformed_gaussians)

    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']

    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()

    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Пропускаем визуализацию для упрощения (можно добавить позже)
    if tracking and visualize_tracking_loss:
        # Здесь можно добавить код, но он требует cam_rots и cam_trans для отладки.
        # Пока пропускаем.
        pass

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    # ... (без изменений)
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution,
                      cam_rots, cam_trans):   # SUBMAP: добавили cam_rots, cam_trans
    # Модифицируем функцию, чтобы использовать переданные cam_rots, cam_trans вместо params['cam_...']
    transformed_gaussians = transform_to_frame_with_cam(params, time_idx, gaussians_grad=False, camera_grad=False,
                                                        cam_rots=cam_rots, cam_trans=cam_trans)   # нужна соответствующая функция
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'], transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    non_presence_mask = non_presence_mask.reshape(-1)

    if torch.sum(non_presence_mask) > 0:
        # Получаем текущую позу камеры из переданных тензоров
        curr_cam_rot = F.normalize(cam_rots[..., time_idx].detach())
        curr_cam_tran = cam_trans[..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def transform_to_frame_with_cam(params, time_idx, gaussians_grad, camera_grad, cam_rots, cam_trans):
    # Аналог transform_to_frame, но с отдельными cam_rots, cam_trans
    if camera_grad:
        cam_rot = F.normalize(cam_rots[..., time_idx])
        cam_tran = cam_trans[..., time_idx]
    else:
        cam_rot = F.normalize(cam_rots[..., time_idx].detach())
        cam_tran = cam_trans[..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    if params['log_scales'].shape[1] == 1:
        transform_rots = False
    else:
        transform_rots = True

    if gaussians_grad:
        pts = params['means3D']
        unnorm_rots = params['unnorm_rotations']
    else:
        pts = params['means3D'].detach()
        unnorm_rots = params['unnorm_rotations'].detach()

    transformed_gaussians = {}
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
    transformed_gaussians['means3D'] = transformed_pts

    if transform_rots:
        from utils.slam_helpers import quat_mult
        norm_rots = F.normalize(unnorm_rots)
        transformed_rots = quat_mult(cam_rot, norm_rots)
        transformed_gaussians['unnorm_rotations'] = transformed_rots
    else:
        transformed_gaussians['unnorm_rotations'] = unnorm_rots

    return transformed_gaussians


def convert_params_to_store(params):
    # ... (без изменений)
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


# ---------------------------------------------------------------------------- #
#                              ОСНОВНАЯ ФУНКЦИЯ                                #
# ---------------------------------------------------------------------------- #

def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False

    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # ------------------------------------------------------------------------ #
    # SUBMAP: Новая инициализация вместо старой
    # ------------------------------------------------------------------------ #
    # Получаем данные первого кадра
    color, depth, intrinsics, pose = dataset[0]
    color = color.permute(2, 0, 1) / 255
    depth = depth.permute(2, 0, 1)
    first_w2c = torch.linalg.inv(pose)
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_w2c.detach().cpu().numpy())

    # Создаём первую подкарту
    first_frame_data = {
        'im': color,
        'depth': depth,
        'intrinsics': intrinsics,
        'w2c': first_w2c,
        'cam': cam
    }
    submap0 = Submap(submap_id=0, first_frame_data, config, num_frames=num_frames)
    submaps = [submap0]
    active_submap = submaps[0]

    # Глобальные тензоры для поз камеры
    global_cam_rots = torch.zeros(1, 4, num_frames).cuda()
    global_cam_trans = torch.zeros(1, 3, num_frames).cuda()
    global_cam_rots[..., 0] = torch.tensor([1, 0, 0, 0], device='cuda')
    global_cam_trans[..., 0] = 0

    # Если есть отдельные датасеты для densification/tracking, пока игнорируем (упрощаем)
    densify_dataset = None
    tracking_dataset = None
    # ------------------------------------------------------------------------ #

    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint (пока не адаптировано, оставляем как есть, но скорее всего не будет использоваться)
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        # TODO: адаптировать загрузку чекпоинта с подкартами
        # Пока пропускаем
        pass
    else:
        checkpoint_time_idx = 0

    # Основной цикл
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Загружаем кадр
        color, depth, _, gt_pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(gt_pose)
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames

        iter_time_idx = time_idx
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_w2c, 'iter_gt_w2c_list': curr_gt_w2c}

        # Инициализация позы камеры для текущего кадра
        if time_idx > 0:
            global_cam_rots, global_cam_trans = initialize_camera_pose_global(
                global_cam_rots, global_cam_trans, time_idx, forward_prop=config['tracking']['forward_prop'])

        # -------------------------------------------------------------------- #
        # Трекинг
        # -------------------------------------------------------------------- #
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Создаём оптимизатор для камеры (только для параметров камеры)
            # В оригинале использовался initialize_optimizer с params, содержащими cam_... параметры.
            # Теперь у нас отдельные тензоры, поэтому создадим оптимизатор вручную.
            # Параметры для оптимизации: global_cam_rots[..., time_idx] и global_cam_trans[..., time_idx]
            # Но они являются частью больших тензоров, поэтому для оптимизации нужно создать отдельные переменные.
            # Проще создать копии для текущего кадра и потом присвоить обратно.
            # Временно оставим старый подход: будем использовать params, но они у нас теперь только в подкарте.
            # Альтернатива: сделать cam_rots и cam_trans параметрами и оптимизировать их напрямую.
            # Для простоты пока создадим отдельные тензоры для текущего кадра и оптимизируем их.
            curr_cam_rot = global_cam_rots[..., time_idx].clone().detach().requires_grad_(True)
            curr_cam_tran = global_cam_trans[..., time_idx].clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([curr_cam_rot, curr_cam_tran], lr=config['tracking']['lrs']['cam_unnorm_rots'])

            candidate_rot = curr_cam_rot.clone().detach()
            candidate_tran = curr_cam_tran.clone().detach()
            current_min_loss = float(1e20)

            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Подставляем текущие оптимизируемые параметры в глобальные тензоры для вызова get_loss_with_cam
                # Создадим временные cam_rots и cam_trans, скопировав глобальные и заменив текущий кадр
                temp_rots = global_cam_rots.clone()
                temp_trans = global_cam_trans.clone()
                temp_rots[..., time_idx] = curr_cam_rot
                temp_trans[..., time_idx] = curr_cam_tran

                loss, new_vars, losses = get_loss_with_cam(
                    active_submap.params, tracking_curr_data, active_submap.variables,
                    iter_time_idx, config['tracking']['loss_weights'],
                    config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                    config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                    tracking=True, cam_rots=temp_rots, cam_trans=temp_trans,
                    plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                    tracking_iteration=iter)
                active_submap.update_variables(new_vars)

                if config['use_wandb']:
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_rot = curr_cam_rot.clone().detach()
                        candidate_tran = curr_cam_tran.clone().detach()

                    if config['report_iter_progress']:
                        # report_progress пока не адаптирован, пропустим
                        pass
                    else:
                        progress_bar.update(1)

                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1

                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Сохраняем лучшую позу
            with torch.no_grad():
                global_cam_rots[..., time_idx] = candidate_rot
                global_cam_trans[..., time_idx] = candidate_tran

        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                global_cam_rots[..., time_idx] = rel_w2c_rot_quat
                global_cam_trans[..., time_idx] = rel_w2c_tran

        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # -------------------------------------------------------------------- #
        # Маппинг
        # -------------------------------------------------------------------- #
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                densify_curr_data = curr_data  # упрощённо
                # Добавляем новые гауссианы
                new_params, new_vars = add_new_gaussians(
                    active_submap.params, active_submap.variables, densify_curr_data,
                    config['mapping']['sil_thres'], time_idx,
                    config['mean_sq_dist_method'], config['gaussian_distribution'],
                    cam_rots=global_cam_rots, cam_trans=global_cam_trans)
                active_submap.params = new_params
                active_submap.variables = new_vars

            # Выбор ключевых кадров (из активной подкарты)
            with torch.no_grad():
                curr_cam_rot = F.normalize(global_cam_rots[..., time_idx].detach())
                curr_cam_tran = global_cam_trans[..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                # Используем ключевые кадры из активной подкарты
                keyframes_for_selection = active_submap.keyframes
                num_keyframes = config['mapping_window_size'] - 2
                # keyframe_selection_overlap ожидает список keyframe_list с полями 'est_w2c', 'depth'
                # У нас в keyframes есть 'w2c', нужно преобразовать.
                # Пока упростим: будем использовать все ключевые кадры подкарты.
                selected_keyframes = list(range(len(keyframes_for_selection)))  # все
                selected_time_idx = [kf['id'] for kf in keyframes_for_selection]
                if len(selected_keyframes) > 0:
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)  # -1 означает текущий кадр
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Оптимизация подкарты
            optimizer = initialize_optimizer(active_submap.params, config['mapping']['lrs'], tracking=False)
            num_iters_mapping = config['mapping']['num_iters']
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Случайный выбор кадра
                rand_idx = np.random.randint(0, len(selected_keyframes))
                sel_idx = selected_keyframes[rand_idx]
                if sel_idx == -1:
                    # текущий кадр
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    kf = keyframes_for_selection[sel_idx]
                    iter_time_idx = kf['id']
                    iter_color = kf['color']
                    iter_depth = kf['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx,
                             'intrinsics': intrinsics, 'w2c': first_w2c, 'iter_gt_w2c_list': iter_gt_w2c}

                loss, new_vars, losses = get_loss_with_cam(
                    active_submap.params, iter_data, active_submap.variables, iter_time_idx,
                    config['mapping']['loss_weights'],
                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                    mapping=True, cam_rots=global_cam_rots, cam_trans=global_cam_trans)
                active_submap.update_variables(new_vars)

                if config['use_wandb']:
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)

                loss.backward()
                with torch.no_grad():
                    if config['mapping']['prune_gaussians']:
                        params, vars_ = prune_gaussians(active_submap.params, active_submap.variables,
                                                         optimizer, iter, config['mapping']['pruning_dict'])
                        active_submap.params = params
                        active_submap.variables = vars_
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, vars_ = densify(active_submap.params, active_submap.variables,
                                                 optimizer, iter, config['mapping']['densify_dict'])
                        active_submap.params = params
                        active_submap.variables = vars_
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if config['report_iter_progress']:
                        # пропускаем визуализацию
                        pass
                    else:
                        progress_bar.update(1)

                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1

            if num_iters_mapping > 0:
                progress_bar.close()
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

        # Добавление ключевого кадра в активную подкарту
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                curr_cam_rot = F.normalize(global_cam_rots[..., time_idx].detach())
                curr_cam_tran = global_cam_trans[..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                active_submap.add_keyframe(time_idx, color, depth, curr_w2c)

        # Чекпоинты пока не адаптированы, пропускаем
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            # TODO: сохранять все подкарты и глобальные позы
            pass

        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    # Вывод среднего времени (без изменений)
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})

    # Финальная оценка (eval) – нужно адаптировать для подкарт. Пока оставляем без изменений, но может не работать.
    # Пропускаем для простоты.

    # Сохранение результатов – пока не реализовано
    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config['seed'])

    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)