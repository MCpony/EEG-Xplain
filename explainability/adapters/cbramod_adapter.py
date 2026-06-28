"""
CBraMod 模型适配器 - 简化版

只负责适配，不负责模型创建和权重加载
"""
import sys
import os
# Add project root to path for importing from parent directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from typing import List, Optional, Callable
from ..model_adapter import ModelAdapter, ModelAdapterRegistry
from ..channel_configs import get_channel_names


# 默认通道名称配置已迁移到 explainability/channel_configs.py
# DEFAULT_CHANNEL_NAMES = {
#     # BCICIV2a - 22 channels (10-20)
#     'bciciv2a': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
#          'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'],
#     # SEEDVIG - 17 channels
#     'seedvig': ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
#          'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2'],
#     # MUMTAZ - 19 channels (10-20)
#     'mumtaz': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
#          'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
#     # PhysioNet - 64 channels (BCI2000)
#     'physionet': ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1',
#          'Cz', 'C2', 'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
#          'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3',
#          'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10',
#          'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
#          'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz'],
#     # STRESS - 20 channels(mental)
#     'stress': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3',
#          'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1',
#          'O2', 'Fz', 'Cz', 'Pz', 'A2-A1'],
#     # Speech - 64 channels (BCI2000)(imaged_speech)
#     'speech': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
#         'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
#         'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10',
#         'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3',
#         'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4',
#         'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'],
#     # SHU - 32 channels (10-20 Extended)
#     'shu': ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
#             'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
#             'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'Oz', 'O2'],
#     # FACED - 32 channels (10-20 Extended)
#     'faced': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
#               'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
#               'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
#               'C6', 'T8'],
# }


@ModelAdapterRegistry.register('cbramod')
class CBraModAdapter(ModelAdapter):
    """
    CBraMod 模型适配器

    用于将 CBraMod 模型适配到可解释性框架

    Example:
        # 1. 用户自己创建和加载模型
        from model_list.cbramod_unified import create_cbramod_model
        model = create_cbramod_model(
            task='tuab',
            foundation_path='foundation.pth',
            checkpoint_path='finetuned.pth',
            device='cuda'
        )

        # 2. 用适配器包装
        adapter = CBraModAdapter(
            model,
            n_channels=16,
            n_patches=10,
            channel_names=['Fp1', 'Fp2', ...]  # 可选
        )

        # 3. 使用可解释性方法
        method = ExplainabilityRegistry.create('gradcam', adapter)
        result = method.explain(input_tensor)
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
