# --------------------------------------------------------
# EEGMamba Adapter for Explainability Framework
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from ..model_adapter import ModelAdapter, ModelAdapterRegistry
from ..channel_configs import get_channel_names
 

@ModelAdapterRegistry.register('eegmamba')
class EEGMambaAdapter(ModelAdapter):
    """
    EEGMamba 模型适配器

    EEGMamba 结构要点：
    - backbone.patch_embedding: PatchEmbedding（时域 Conv + 频域 FFT → 嵌入）
    - backbone.encoder: MixerModel（12 层 Mamba2，逐层 flip 实现双向扫描）
    - classifier: 各 task 的分类头
    - 输入: (B, C, L, P)，C=通道数, L=patch数, P=patch_size(200)
    - backbone 输出: (B, C, L, 200)
    """

    model_name = "eegmamba"
    supported_methods = ['gradcam', 'ig', 'gradient_shap', 'saliency', 'occlusion']
    supports_double = False  # Mamba2 CUDA kernels only support float32/float16/bfloat16

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
            model:         已创建并加载好权重的 EEGMamba 下游模型
            config:        对应 task 的配置字典（来自 configs/eegmamba.yaml）
                           必须包含 n_channels, n_patches, num_classes
            task:          任务名称（用于从 channel_configs 获取通道名称）
            channel_names: 通道名称列表（可选，优先级最高）
            device:        计算设备
        """
        super().__init__(model, device)

        self._n_channels = config['n_channels']
        self._n_patches = config['n_patches']
        self._patch_size = 200  # EEGMamba 固定 patch_size=200
        self._d_model = 200     # EEGMamba 固定 d_model=200

        num_classes = config.get('num_classes', None)
        is_binary = config.get('is_binary', False)
        self.is_binary = is_binary or (num_classes == 1)

        # 通道名称：手动指定 > task 配置 > 通用命名
        if channel_names is not None:
            self._channel_names = channel_names
        else:
            self._channel_names = get_channel_names('eegmamba', task, self._n_channels)

        self._target_layer = self._find_target_layer()

    def _find_target_layer(self) -> nn.Module:
        """
        查找 GradCAM 目标层：backbone.encoder.norm_f（RMSNorm）。
        """
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            if hasattr(backbone, 'encoder'):
                encoder = backbone.encoder
                if hasattr(encoder, 'norm_f'):
                    return encoder.norm_f
        # 回退：遍历搜索
        for name, module in self.model.named_modules():
            if 'encoder' in name and 'norm_f' in name:
                return module
        return self.model

    def get_target_layer(self) -> nn.Module:
        return self._target_layer

    def get_n_channels(self) -> int:
        return self._n_channels

    def get_n_patches(self) -> int:
        return self._n_patches

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def get_input_shape(self) -> Tuple[int, ...]:
        """EEGMamba 期望输入: (B, C, L, patch_size)"""
        return (1, self._n_channels, self._n_patches, self._patch_size)

    def get_reshape_transform(self) -> Optional[Callable]:
        """
        GradCAM reshape 变换。
        norm_f 输出为 (B, C*L, d_model)，
        reshape 为 (B, d_model, C, L) 供 GradCAM 计算空间热力图。
        GradCAM 对 d_model 维做 global average pooling，最终热力图为 (B, C, L)。
        """
        n_channels = self._n_channels
        n_patches = self._n_patches

        def reshape_fn(tensor):
            # tensor: (B, C*L, d_model)
            if tensor.dim() == 3:
                B, CL, D = tensor.shape
                tensor = tensor.view(B, n_channels, n_patches, D)
                tensor = tensor.permute(0, 3, 1, 2)  # (B, D, C, L)
            return tensor

        return reshape_fn

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """
        预处理输入数据。
        Args:
            x: numpy array，形状 (C, L, P) 或 (B, C, L, P)
               C=通道数, L=patch数, P=patch_size(200)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (C, L, P) -> (1, C, L, P)
        if x.shape[1] != self._n_channels:
            raise ValueError(
                f"Input channels {x.shape[1]} != expected {self._n_channels}"
            )
        if x.shape[2] != self._n_patches:
            raise ValueError(
                f"Input n_patches {x.shape[2]} != expected {self._n_patches}"
            )
        return x.to(self.device)

    def to_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """将输入转为 (B, C, n_patches, patch_size) 格式。
        处理两种情况：
        - 已经是 4D (B, C, L, P): 直接返回
        - 3D (B, C, T): 按 patch_size 切分为 (B, C, T//patch_size, patch_size)
        """
        if x.dim() == 4:
            return x
        if x.dim() == 3:
            B, C, T = x.shape
            return x.reshape(B, C, T // self._patch_size, self._patch_size)
        return x.unsqueeze(0)

    def from_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """EEGMamba 模型直接接受 4D patch 输入，无需还原"""
        return x

    def postprocess_attribution(self, attribution: np.ndarray) -> np.ndarray:
        """后处理归因结果，去除 batch 维度"""
        if attribution.ndim == 4:
            attribution = attribution.squeeze(0)
        return attribution

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'patch_size': self._patch_size,
            'd_model': self._d_model,
            'target_layer': str(self._target_layer),
            'architecture': 'mamba2_bidirectional',
            'n_mamba_layers': 12,
        })
        return info
