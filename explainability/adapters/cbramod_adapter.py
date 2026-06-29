
import sys
import os
# Add project root to path for importing from parent directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from typing import List, Optional, Callable
from ..model_adapter import ModelAdapter, ModelAdapterRegistry
from ..channel_configs import get_channel_name


@ModelAdapterRegistry.register('cbramod')
class CBraModAdapter(ModelAdapter):
    """
    CBraMod 模型适配器

    用于将 CBraMod 模型适配到可解释性框架

    """ 

    model_name = "cbramod"
    supported_methods = ['gradcam', 'ig', 'gradient_shap', 'saliency',
                         'occlusion', 'lime', 'shap']

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        #n_channels: int,
        #n_patches: int,
        task: Optional[str] = None,
        channel_names: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 已创建并加载好权重的CBraMod模型
            n_channels: EEG通道数
            n_patches: 时间patch数
            task: 任务名称（用于获取正确的通道名称）
            channel_names: 通道名称列表（可选，优先级最高）
            device: 计算设备
        """
        super().__init__(model, device)

        self._n_channels = config['n_channels'] if 'n_channels' in config else None
        self._n_patches = config['n_patches'] if 'n_patches' in config else None

        # 判断是否二分类（num_classes==1 → sigmoid）
        num_classes = config.get('num_classes', None)
        if num_classes is None and hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        self.is_binary = (num_classes == 1)

        # 设置通道名称（优先级：手动指定 > task配置 > 通用命名）
        if channel_names is not None:
            self._channel_names = channel_names
        else:
            self._channel_names = get_channel_names('cbramod', task, self._n_channels)

        # 查找目标层
        self._target_layer = self._find_target_layer()

    def _find_target_layer(self) -> nn.Module:
        """查找目标层用于GradCAM"""
        # CBraMod 结构: model.backbone.encoder.layers[-1]
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            if hasattr(backbone, 'encoder'):
                encoder = backbone.encoder
                if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
                    return encoder.layers[-1]
                return encoder

        # 回退：搜索所有模块
        for name, module in self.model.named_modules():
            if 'encoder' in name.lower() and 'layer' in name.lower():
                return module

        return self.model

    def get_target_layer(self) -> nn.Module:
        """获取GradCAM目标层"""
        return self._target_layer

    def get_n_channels(self) -> int:
        return self._n_channels

    def get_n_patches(self) -> int:
        return self._n_patches

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def set_channel_names(self, channel_names: List[str]):
        """设置通道名称"""
        if len(channel_names) != self._n_channels:
            raise ValueError(f"Channel names count ({len(channel_names)}) != n_channels ({self._n_channels})")
        self._channel_names = channel_names

    def get_reshape_transform(self) -> Optional[Callable]:
        """
        获取GradCAM的reshape变换

        CBraMod的输出形状是 (batch, channels, patches, features)
        需要转换为 (batch, features, channels, patches) 用于GradCAM
        """
        def reshape_transform(tensor):
            if tensor.dim() == 4:
                return tensor.permute(0, 3, 1, 2)
            return tensor

        return reshape_transform

    def from_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """CBraMod forward 期望 4D (B, C, N, patch_size)，不做 flatten。"""
        return x

    def prepare_input(self, x) -> torch.Tensor:
        """CBraMod 训练时 __getitem__ 做了 /100，推理和归因时需保持一致。"""
        import numpy as np
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = self.to_patch_input(x)
        x = self.from_patch_input(x)
        x = x / 100.0
        return x
