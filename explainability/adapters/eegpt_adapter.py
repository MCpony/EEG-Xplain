# --------------------------------------------------------
# EEGPT Adapter for Explainability Framework
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from ..model_adapter import ModelAdapter, ModelAdapterRegistry
from ..channel_configs import get_channel_names
 
# EEGPT standard channel dictionary (mirrors CHANNEL_DICT in EEGPT source)
EEGPT_CHANNEL_DICT = {k.upper(): v for v, k in enumerate(
    ['FP1', 'FPZ', 'FP2',
     'AF7', 'AF3', 'AF4', 'AF8',
     'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
     'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
     'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
     'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
     'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
     'O1', 'OZ', 'O2']
)}


@ModelAdapterRegistry.register('eegpt')
class EEGPTAdapter(ModelAdapter):
    """
    EEGPT 模型适配器

    支持两种 EEGPT 微调模式：
    - linear_probe: target_encoder + reconstructor/predictor，head 处理 cls token 均值
    - full_finetune: 只用 target_encoder，输出 flatten 后直接接 head（tuab/tuev 使用）

    模型结构要点：
    - target_encoder.blocks: 8层 Transformer Block
    - 输入: (B, C, T)，经 patch_embed 后为 (B, N_patches, N_channels, embed_dim)
    - 每个 patch 内部：(N_channels + embed_num) tokens 经 Attention
    - embed_dim=512, embed_num=4, patch_size=64

    """

    model_name = "eegpt"
    supported_methods = ['gradcam', 'ig', 'gradient_shap', 'saliency',
                         'occlusion', 'lime', 'shap']

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        task: Optional[str] = None,
        channel_names: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 已创建并加载好权重的 EEGPT 模型 (EEGPTClassifier)
            config: 任务配置字典，需包含 'in_channels'、'img_size' 和 'mode' 字段
                    mode: 'linear_probe' 或 'full_finetune'
            task: 任务名称（用于获取通道名称）
            channel_names: 通道名称列表（可选，优先级最高）
            device: 计算设备
        """
        super().__init__(model, device)
        self._config = config
        model_config = config.get('model', config)  # 兼容分层和扁平两种结构

        self.is_binary = (model_config.get('num_classes', 2) == 1)

        # 原始输入通道数（数据真实电极数，chan_conv 的输入维度）
        self._in_channels = model_config['in_channels']
        # encoder 通道数（chan_conv 输出/img_size[0]，用于 GradCAM reshape 切 token）
        self._enc_channels = model_config['img_size'][0]
        self._mode = config.get('mode', 'full_finetune')

        # patch_size=64 固定，从 img_size 推算 n_patches
        img_size = model_config.get('img_size', None)
        if img_size is not None:
            self._n_patches = img_size[1] // 64
        else:
            self._n_patches = None

        # chan_conv 在无 padding 时吃掉的时间点数（用于 to_patch_input 正确截断）
        chan_conv_kernel = model_config.get('chan_conv_kernel', None)
        use_chan_conv = model_config.get('use_chan_conv', False)
        if use_chan_conv and chan_conv_kernel is not None and chan_conv_kernel == 15:
            self._chan_conv_time_loss = chan_conv_kernel - 1
        else:
            self._chan_conv_time_loss = 0

        # embed_num=4 固定（由预训练权重决定）
        self._embed_num = 4
        self._embed_dim = 512

        # 通道名对应原始输入电极，用 eegpt-{task} key 从 channel_configs 取
        if channel_names is not None:
            self._channel_names = channel_names
        else:
            self._channel_names = get_channel_names('eegpt', task, self._in_channels)

        # physio_p300: 原始数据 64 通道，preprocess 里需切片到 58 通道
        if task == 'physio_p300':
            _all_ch = [x.upper() for x in [
                'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
                'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
                'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
                'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
                'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
                'P10', 'PO8', 'PO4', 'O2'
            ]]
            self._channels_index = [_all_ch.index(n) for n in self._channel_names if n in _all_ch]
        else:
            self._channels_index = None

        # 是否在预处理阶段做时间插值
        self._temporal_interpolation = config.get('temporal_interpolation', False)

        # 查找目标层
        self._target_layer = self._find_target_layer()

    def _find_target_layer(self) -> nn.Module:
        """查找用于 GradCAM 的目标层
        linear_probe 模式下 forward() 会临时给 target_encoder 开梯度，
        所以可以直接用 target_encoder 最后一个 block 作为目标层，
        能得到完整的通道×时间热力图。
        """
        if hasattr(self.model, 'target_encoder'):
            enc = self.model.target_encoder
            if hasattr(enc, 'blocks') and len(enc.blocks) > 0:
                return enc.blocks[-1]

        if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
            return self.model.blocks[-1]

        return self.model

    def get_target_layer(self) -> nn.Module:
        return self._target_layer

    def prepare_input(self, x) -> torch.Tensor:
        """EEGPT 模型期望原始 (B, C, T) 输入，跳过 patch 截断的 roundtrip"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def to_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """将 3D (B, C, T) 切分为 4D (B, C, N_patches, patch_size) 供 LIME/Occlusion 使用

        保存原始时间长度，from_patch_input 会还原完整 T（含尾部未对齐的采样点）。
        """
        if x.dim() == 4:
            return x
        B, C, T = x.shape
        patch_size = 64
        N = T // patch_size
        self._original_T = T
        self._tail = x[:, :, N * patch_size:]  # 尾部不足一个 patch 的部分
        return x[:, :, :N * patch_size].reshape(B, C, N, patch_size)

    def from_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """将 4D patch 格式 (B, C, N, patch_size) 还原为 3D (B, C, T)

        拼回 to_patch_input 时保存的尾部采样点，保证输出 T 与原始输入一致。
        """
        if x.dim() == 3:
            return x
        B, C, N, ps = x.shape
        main = x.reshape(B, C, N * ps)
        if hasattr(self, '_tail') and self._tail is not None and self._tail.shape[-1] > 0:
            tail = self._tail.expand(B, -1, -1).to(main.device)
            return torch.cat([main, tail], dim=-1)
        return main

    def get_n_channels(self) -> int:
        return self._in_channels

    def get_n_patches(self) -> Optional[int]:
        return self._n_patches

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（直接调用模型，chan_ids 已在模型内部存储）"""
        if self._mode == 'linear_probe' and hasattr(self.model, 'target_encoder'):
            for p in self.model.target_encoder.parameters():
                p.requires_grad_(True)
        if x.dim() == 4:
            x = x.flatten(2)
        return self.model(x)

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        返回期望输入形状: (batch, in_channels, time_steps)
        in_channels 可能与 n_channels 不同（use_chan_conv=True 时会做映射）
        """
        n_time = self._n_patches * 64 if self._n_patches else None
        return (1, self._in_channels, n_time)

    def get_reshape_transform(self) -> Optional[Callable]:
        """
        linear_probe 模式目标层为 linear_probe1，输出 (B, N_patches, lp1_out)，
        """
        # target_encoder.blocks[-1] 输出 (B*N_patches, C+embed_num, D)
        # linear_probe 和 full_finetune 都用同一个 reshape
        n_channels = self._enc_channels
        n_patches = self._n_patches

        def reshape_fn(tensor):
            if tensor.dim() == 3:
                BN, _, D = tensor.shape
                if n_patches is not None:
                    N = n_patches
                    B = BN // N
                else:
                    B = 1
                    N = BN
                chan_tokens = tensor[:, :n_channels, :]       # (B*N, C, D)
                chan_tokens = chan_tokens.view(B, N, n_channels, D)
                chan_tokens = chan_tokens.permute(0, 3, 2, 1)  # (B, D, C, N)
                return chan_tokens
            return tensor

        return reshape_fn

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """预处理输入：numpy array -> tensor，确保维度正确"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 2:
            # (C, T) -> (1, C, T)
            x = x.unsqueeze(0)
        # physio_p300: 原始 64 通道切片到 58 通道
        if self._channels_index is not None:
            x = x[:, self._channels_index, :]
        if x.dim() == 3 and x.shape[1] != self._in_channels:
            raise ValueError(
                f"Input channels {x.shape[1]} doesn't match expected {self._in_channels}"
            )
        if self._temporal_interpolation:
            model_config = self._config.get('model', self._config)
            target_len = model_config['img_size'][1]
            if x.shape[-1] != target_len:
                x = torch.nn.functional.interpolate(x, size=target_len, mode='nearest')
        return x.to(self.device)

    def postprocess_attribution(self, attribution: np.ndarray) -> np.ndarray:
        """后处理归因结果"""
        if attribution.ndim == 4:
            attribution = attribution.squeeze(0)
        return attribution

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'mode': self._mode,
            'embed_dim': self._embed_dim,
            'embed_num': self._embed_num,
            'target_layer': str(self._target_layer),
            'architecture': 'transformer',
            'patch_size': 64,
        })
        return info
