"""
Occlusion (遮挡) 方法实现
"""
import torch
import numpy as np
from typing import Dict, Any, Optional
from ..base import ExplainabilityMethod, ExplainabilityRegistry


@ExplainabilityRegistry.register('occlusion')
class OcclusionMethod(ExplainabilityMethod):
    """
    Occlusion (遮挡敏感性分析)

    通过遮挡输入的不同部分来评估其重要性
    """

    name = "occlusion"
    description = "Occlusion - 通过遮挡分析特征重要性，直观但计算量大"

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda',
                 window_size: tuple = (1, 1, 50), stride: tuple = (1, 1, 25),
                 baseline_value: float = 0.0):
        """
        Args:
            model_adapter: 模型适配器
            device: 计算设备
            window_size: 遮挡窗口大小 (channels, patches, features)
            stride: 滑动步长
            baseline_value: 遮挡区域的填充值
        """
        super().__init__(model_adapter, device)
        self.window_size = window_size
        self.stride = stride
        self.baseline_value = baseline_value

    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        计算Occlusion归因

        Args:
            input_tensor: 输入张量
            target: 目标类别

        Returns:
            归因结果字典
        """
        input_tensor = input_tensor.to(self.device)

        # 获取原始预测
        with torch.no_grad():
            original_output = self.model_adapter.forward(input_tensor)
            if target is None:
                if original_output.numel() == 1:
                    original_pred = original_output.item()
                else:
                    target = original_output.argmax(dim=-1).item()
                    original_pred = original_output[0, target].item() if original_output.dim() > 1 else original_output[target].item()
            else:
                if original_output.numel() == 1:
                    original_pred = original_output.item() if target != 0 else -original_output.item()
                elif original_output.dim() > 1:
                    original_pred = original_output[0, target].item()
                else:
                    original_pred = original_output[target].item()

        # 获取输入形状（转为 patch 化的 4D 格式）
        patch_input = self.model_adapter.to_patch_input(input_tensor)
        batch, n_channels, n_patches, n_features = patch_input.shape
        w_ch, w_p, w_f = self.window_size
        s_ch, s_p, s_f = self.stride

        # 计算归因图大小
        attr_ch = (n_channels - w_ch) // s_ch + 1
        attr_p = (n_patches - w_p) // s_p + 1

        # 存储重要性
        importance = np.zeros((attr_ch, attr_p))
        counts = np.zeros((attr_ch, attr_p))

        # 滑动窗口遮挡
        with torch.no_grad():
            for i, ch_start in enumerate(range(0, n_channels - w_ch + 1, s_ch)):
                for j, p_start in enumerate(range(0, n_patches - w_p + 1, s_p)):
                    # 创建遮挡版本
                    occluded = patch_input.clone()
                    # 用该通道的均值替代，避免零值被 LayerNorm 补偿
                    for ch in range(ch_start, ch_start + w_ch):
                        ch_mean = patch_input[0, ch].mean()
                        occluded[0, ch, p_start:p_start+w_p, :] = ch_mean

                    # 还原为模型输入格式再预测
                    occluded = self.model_adapter.from_patch_input(occluded)
                    output = self.model_adapter.forward(occluded)
                    if target is None:
                        if output.numel() == 1:
                            pred = output.item()
                        else:
                            pred = output.max().item()
                    else:
                        if output.numel() == output.shape[0]:
                            raw = output.squeeze(-1).cpu() if output.dim() > 1 else output
                            pred = raw.item() if target != 0 else -raw.item()
                        elif output.dim() > 1:
                            pred = output[0, target].item()
                        else:
                            pred = output[target].item()

                    # 计算预测差异（差异越大说明该区域越重要）
                    importance[i, j] = original_pred - pred
                    counts[i, j] += 1
                    del occluded, output

        # 处理counts为0的情况
        counts[counts == 0] = 1
        importance = importance / counts

        # 将归因图上采样到原始通道×patch大小
        attr = self._upsample_attribution(importance, n_channels, n_patches)

        # 归一化
        attr = attr / (np.abs(attr).max() + 1e-8)

        spatial = np.mean(attr, axis=-1)
        print(f"[DEBUG Occlusion] raw importance: min={importance.min():.4f}, max={importance.max():.4f}, "
              f"mean={importance.mean():.4f}")
        print(f"[DEBUG Occlusion] attr (C,N): min={attr.min():.4f}, max={attr.max():.4f}")
        print(f"[DEBUG Occlusion] spatial_importance: min={spatial.min():.4f}, max={spatial.max():.4f}, "
              f"pos={np.sum(spatial>0)}, neg={np.sum(spatial<0)}")
        print(f"[DEBUG Occlusion] original_pred={original_pred:.4f}")

        return {
            'combined': attr,
            'spatial_importance': spatial,
            'temporal_importance': np.mean(attr, axis=0),
            'raw_importance': importance,
            'window_size': self.window_size,
            'stride': self.stride,
        }

    def _upsample_attribution(self, importance: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """上采样归因图到目标大小"""
        from scipy.ndimage import zoom

        h, w = importance.shape
        zoom_h = target_h / h
        zoom_w = target_w / w

        return zoom(importance, (zoom_h, zoom_w), order=1)

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'window_size': (1, 1, 50),
            'stride': (1, 1, 25),
            'baseline_value': 0.0,
        }
