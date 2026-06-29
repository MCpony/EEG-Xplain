# --------------------------------------------------------
# LaBraM Adapter for Explainability Framework
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from ..model_adapter import ModelAdapter, ModelAdapterRegistry
from ..channel_configs import get_channel_names
 
#channel_names = ['FP1', 'FP2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']
standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


@ModelAdapterRegistry.register('labram')
class LaBraMAdapter(ModelAdapter):


    model_name = "labram"
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
            model: 已创建并加载好权重的LaBraM模型
            n_channels: EEG通道数
            task: 任务名称（用于获取正确的通道名称）
            channel_names: 通道名称列表（可选，优先级最高）
            device: 计算设备
        """
        super().__init__(model, device)

        self.is_binary = (config.get('num_classes', 2) == 1)
        self._n_channels = config['n_channels']
        self._patch_size = 200  # LaBraM 固定使用 200

        # 设置通道名称（优先级：手动指定 > task配置 > 通用命名）
        if channel_names is not None:
            self._channel_names = channel_names
        else:
            self._channel_names = get_channel_names('labram', task, self._n_channels)

        # 预计算通道索引（用于 pos_embed 选择）
        self._input_chans = self.get_input_chans(self._channel_names)

        # 查找目标层
        self._target_layer = self._find_target_layer()

    def get_input_chans(self,ch_names):
        global standard_1020
        lower_map = {ch.lower(): i for i, ch in enumerate(standard_1020)}
        input_chans = [0] # for cls token
        for ch_name in ch_names:
            input_chans.append(lower_map[ch_name.lower()] + 1)
        return torch.tensor(input_chans, dtype=torch.long)


    def _find_target_layer(self) -> nn.Module:
        """查找目标层用于GradCAM
        LaBraM 结构：blocks.0, blocks.1, ..., blocks.11, fc_norm, head
        选择最后一个 Transformer block
        """
        if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:  # 判断模型是否存在 blocks 属性，并且 blocks 列表长度大于 0
            return self.model.blocks[-1]  # 返回最后一个 blocks 模块
        # 回退：搜索所有模块
        target_layer = None  # 初始化目标层为 None
        for name, module in self.model.named_modules():  # 遍历模型中的所有模块
            if 'block' in name.lower() and isinstance(module, nn.Module):  # 判断模块名称是否包含 "block"，并且模块是 nn.Module 类型
                target_layer = module  # 将模块作为目标层返回
        return target_layer or self.model  # 如果没有找到满足条件的模块，则返回模型本身

    def get_target_layer(self) -> nn.Module:
        """返回用于梯度计算的目标层"""
        return self._target_layer

    def get_channel_names(self) -> List[str]:
        """返回 EEG 通道名称"""
        return self._channel_names

    def get_n_channels(self) -> int:
        """返回通道数"""
        return self._n_channels

    def get_n_patches(self) -> Optional[int]:
        """返回时间patch数（LaBraM 动态推算，返回 None）"""
        return None

    def to_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """LaBraM patch_size=200，n_patches 从 T 动态计算。"""
        if x.dim() == 4:
            return x
        B, C, T = x.shape
        n_patches = T // self._patch_size
        return x[:, :, :n_patches * self._patch_size].reshape(B, C, n_patches, self._patch_size)

    def from_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """LaBraM forward 期望 4D (B, n_electrodes, n_patches, patch_size)，不做 flatten。"""
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """重写 forward，传入 input_chans 以选择正确的 pos_embed 位置"""
        input_chans = self._input_chans.to(x.device)
        return self.model(x, input_chans=input_chans)

    def get_input_shape(self) -> Tuple[int, ...]:
        """获取模型期望的输入形状（n_patches 动态推算，此处返回 None 占位）"""
        return (1, self._n_channels, None, self._patch_size)

    def get_reshape_transform(self) -> Optional[Callable]:
        """
        LaBraM 是 Transformer 架构，GradCAM 需要将输出 reshape 回空间维度
        """
        def reshape_fn(tensor):
            # tensor shape: (batch, seq_len, embed_dim)
            if tensor.dim() == 3:
                batch_size, seq_len, _ = tensor.shape
                # 去掉 CLS token，动态推算 n_patches
                n_patches = (seq_len - 1) // self._n_channels
                tensor = tensor[:, 1:, :]
                tensor = tensor.view(batch_size, self._n_channels, n_patches, -1)
                # GradCAM 期望 (batch, embed_dim, channels, patches)
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor

        return reshape_fn

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """预处理输入数据"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加 batch 维度
        if x.shape[1] != self._n_channels:
            raise ValueError(f"Input channels {x.shape[1]} doesn't match expected {self._n_channels}")
        return x.to(self.device)

    def postprocess_attribution(self, attribution: np.ndarray) -> np.ndarray:
        """后处理归因结果"""
        if attribution.ndim == 4:
            attribution = attribution.squeeze(0)  # 移除 batch 维度
        return attribution

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        info = super().get_model_info()
        info.update({
            'patch_size': self._patch_size,
            'target_layer': str(self._target_layer),
            'architecture': 'transformer'
        })
        return info
