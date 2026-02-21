import torch
import numpy as np
from utils.slam_external import build_rotation
from utils.slam_helpers import transform_to_frame, transformed_params2rendervar, transformed_params2depthplussilhouette
from utils.keyframe_selection import keyframe_selection_overlap

class Submap:
    def __init__(self, submap_id, first_frame_data, config, num_frames_global):
        self.id = submap_id
        self.keyframes = []          # список ключевых кадров (словари с id, color, depth, w2c)
        self.keyframe_indices = []   # глобальные индексы кадров
        self.params = None            # словарь параметров гауссианов (как в оригинале)
        self.variables = None         # словарь переменных (max_2D_radius, etc.)
        self.first_frame_w2c = first_frame_data['w2c'].clone()  # поза первого кадра подкарты
        self.intrinsics = first_frame_data['intrinsics'].clone()
        self.cam = first_frame_data['cam']  # объект камеры
        self.optimizer = None
        self.num_frames_global = num_frames_global  # общее число кадров (для инициализации поз камеры)

        # Инициализация подкарты первым кадром
        self.initialize_from_first_frame(first_frame_data, config)

    def initialize_from_first_frame(self, frame_data, config):
        """Инициализирует params и variables из первого кадра (аналог initialize_first_timestep)."""
        # Используем существующую функцию initialize_first_timestep, но адаптируем
        # Вместо возврата params, variables, мы сохраняем их в self.
        # Примечание: initialize_first_timestep ожидает dataset, но у нас только один кадр.
        # Мы скопируем логику, но без dataset.
        color = frame_data['im']  # уже тензор (C,H,W)
        depth = frame_data['depth']  # тензор (C,H,W)
        intrinsics = frame_data['intrinsics']
        w2c = frame_data['w2c']

        # Get Initial Point Cloud
        mask = (depth > 0).reshape(-1)
        init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c,
                                                     mask=mask, compute_mean_sq_dist=True,
                                                     mean_sq_dist_method=config['mean_sq_dist_method'])

        # Initialize parameters (аналогично initialize_params, но без камерных траекторий)
        num_pts = init_pt_cld.shape[0]
        means3D = init_pt_cld[:, :3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        gaussian_distribution = config['gaussian_distribution']
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
        # Преобразуем в параметры
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        variables = {
            'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'scene_radius': torch.max(depth) / config['scene_radius_depth_ratio']
        }

        self.params = params
        self.variables = variables

    def add_keyframe(self, frame_id, color, depth, w2c):
        """Добавляет ключевой кадр в подкарту."""
        self.keyframes.append({'id': frame_id, 'color': color, 'depth': depth, 'w2c': w2c})
        self.keyframe_indices.append(frame_id)

    def get_params_for_frame(self, time_idx):
        """Возвращает params для использования в трекинге/маппинге (с учётом камерных параметров)."""
        # В SplaTAM params содержит cam_unnorm_rots и cam_trans для всех кадров глобально.
        # При переходе на подкарты мы должны либо хранить позы камер глобально отдельно,
        # либо включить их в подкарту. Лучше хранить глобально, так как трекинг использует их.
        # Но при маппинге мы будем использовать только params гауссианов подкарты.
        return self.params

    def get_variables(self):
        return self.variables

    def update_variables(self, new_vars):
        self.variables.update(new_vars)

    def create_optimizer(self, lrs_dict, tracking=False):
        """Создаёт оптимизатор для параметров подкарты."""
        from scripts.splatam import initialize_optimizer  # импорт внутри метода
        self.optimizer = initialize_optimizer(self.params, lrs_dict, tracking)
        return self.optimizer

    def save_checkpoint(self, path):
        """Сохраняет состояние подкарты в файл."""
        torch.save({
            'params': {k: v.detach().cpu() for k, v in self.params.items()},
            'variables': self.variables,
            'keyframes': self.keyframes,
            'keyframe_indices': self.keyframe_indices,
            'first_frame_w2c': self.first_frame_w2c.cpu(),
            'intrinsics': self.intrinsics.cpu(),
            'id': self.id
        }, path)

    def load_checkpoint(self, path):
        """Загружает состояние подкарты."""
        data = torch.load(path)
        self.params = {k: torch.nn.Parameter(v.cuda().float().requires_grad_(True)) for k, v in data['params'].items()}
        self.variables = data['variables']
        self.keyframes = data['keyframes']
        self.keyframe_indices = data['keyframe_indices']
        self.first_frame_w2c = data['first_frame_w2c'].cuda()
        self.intrinsics = data['intrinsics'].cuda()
        self.id = data['id']