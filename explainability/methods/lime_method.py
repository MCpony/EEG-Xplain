"""
LIME (Local Interpretable Model-agnostic Explanations) 实现
"""
import torch
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from ..base import ExplainabilityMethod, ExplainabilityRegistry


@ExplainabilityRegistry.register('lime')
class LIMEMethod(ExplainabilityMethod):
    """
    LIME (Local Interpretable Model-agnostic Explanations)

    通过局部线性近似解释模型预测
    """

    name = "lime"
    description = "LIME - 局部可解释模型，通过扰动生成解释"

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda',
                 n_samples: int = 1000, kernel_width: float = 1.0,
                 unit: str = 'channel_patch'):
        """
        Args:
            model_adapter: 模型适配器
            device: 计算设备
            n_samples: 扰动样本数量
            kernel_width: 核函数宽度（越大权重衰减越慢）
            unit: 扰动单位 ('channel', 'patch', 'channel_patch')
        """
        super().__init__(model_adapter, device)
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.unit = unit

    def _generate_perturbations(self, input_tensor: torch.Tensor) -> tuple:
        """
        生成扰动样本

        Returns:
            (perturbed_inputs, perturbation_masks)
        """
        batch_size, n_channels, n_patches, n_features = input_tensor.shape
        self._last_n_patches = n_patches
        self._last_n_channels = n_channels

        if self.unit == 'channel':
            # 按通道扰动
            n_units = n_channels
        elif self.unit == 'patch':
            # 按时间patch扰动
            n_units = n_patches
        else:  # channel_patch
            # 按通道×patch扰动
            n_units = n_channels * n_patches

        # 生成二进制掩码
        masks = np.random.binomial(1, 0.5, size=(self.n_samples, n_units)).astype(np.float32)

        # 生成扰动样本
        perturbed_inputs = []
        for mask in masks:
            perturbed = input_tensor.clone()

            if self.unit == 'channel':
                for ch in range(n_channels):
                    if mask[ch] == 0:
                        perturbed[0, ch] = 0  # 将该通道置零
            elif self.unit == 'patch':
                for p in range(n_patches):
                    if mask[p] == 0:
                        perturbed[0, :, p] = 0  # 将该patch置零
            else:  # channel_patch
                idx = 0
                for ch in range(n_channels):
                    for p in range(n_patches):
                        if mask[idx] == 0:
                            perturbed[0, ch, p] = 0
                        idx += 1

            perturbed_inputs.append(perturbed)

        perturbed_inputs = torch.cat(perturbed_inputs, dim=0)
        return perturbed_inputs, masks

    def _compute_kernel_weights(self, masks: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
        """计算核函数权重"""
        # 使用指数核
        distances = np.sqrt(np.sum((masks - original_mask) ** 2, axis=1))

        # 归一化距离以避免权重全为0
        if distances.max() > 0:
            distances = distances / distances.max()

        # 使用更大的 kernel_width 避免权重过小
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # 确保权重和不为0，添加小的正则项
        if weights.sum() < 1e-10:
            weights = np.ones_like(weights)  # 如果权重全为0，使用均匀权重

        return weights

    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        计算LIME归因

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
                    original_pred = original_output.max().item()
            else:
                if original_output.numel() == 1:
                    # 二分类单 logit：直接用输出值，class 0 取反
                    original_pred = original_output.item() if target != 0 else -original_output.item()
                elif original_output.dim() > 1:
                    original_pred = original_output[0, target].item()
                else:
                    original_pred = original_output[target].item()

        # 生成扰动
        patch_input = self.model_adapter.to_patch_input(input_tensor)
        perturbed_inputs, masks = self._generate_perturbations(patch_input)

        # 批量预测扰动样本
        predictions = []
        batch_size = 100  # 批处理大小
        with torch.no_grad():
            for i in range(0, len(perturbed_inputs), batch_size):
                batch = perturbed_inputs[i:i+batch_size].to(self.device)
                batch = self.model_adapter.from_patch_input(batch)
                output = self.model_adapter.forward(batch)
                if target is None:
                    if output.dim() == 1:
                        preds = output.cpu().numpy()
                    else:
                        preds = output.max(dim=-1)[0].cpu().numpy()
                else:
                    if output.numel() == output.shape[0]:
                        raw = output.squeeze(-1).cpu().numpy()
                        preds = raw if target != 0 else -raw
                    else:
                        preds = output[:, target].cpu().numpy()
                predictions.extend(preds.flatten())
                del batch, output

        predictions = np.array(predictions)

        # 计算核权重
        original_mask = np.ones(masks.shape[1])
        weights = self._compute_kernel_weights(masks, original_mask)

        # 使用加权岭回归拟合局部线性模型
        model = Ridge(alpha=1.0)
        model.fit(masks, predictions, sample_weight=weights)

        # 系数即为特征重要性
        coefficients = model.coef_

        # 重构为(channels, patches)形状 — 使用实际扰动时的维度
        n_channels = getattr(self, '_last_n_channels', self.model_adapter.get_n_channels())
        n_patches = getattr(self, '_last_n_patches', self.model_adapter.get_n_patches())

        if self.unit == 'channel':
            # (n_channels,) -> (n_channels, n_patches)
            attr = np.tile(coefficients.reshape(-1, 1), (1, n_patches))
        elif self.unit == 'patch':
            # (n_patches,) -> (n_channels, n_patches)
            attr = np.tile(coefficients.reshape(1, -1), (n_channels, 1))
        else:  # channel_patch
            # (n_channels * n_patches,) -> (n_channels, n_patches)
            attr = coefficients.reshape(n_channels, n_patches)

        # 保留符号，以绝对值最大值归一化到 [-1, 1]
        attr_max = np.abs(attr).max()
        if attr_max > 1e-8:
            attr = attr / attr_max
        else:
            print("Warning: LIME coefficients are all near zero. Model may be too complex or perturbations ineffective.")

        spatial = np.mean(attr, axis=-1)
        print(f"[DEBUG LIME] coefficients raw: min={coefficients.min():.4f}, max={coefficients.max():.4f}, "
              f"mean={coefficients.mean():.4f}")
        print(f"[DEBUG LIME] attr (C,N): min={attr.min():.4f}, max={attr.max():.4f}")
        print(f"[DEBUG LIME] spatial_importance: min={spatial.min():.4f}, max={spatial.max():.4f}, "
              f"pos={np.sum(spatial>0)}, neg={np.sum(spatial<0)}")
        print(f"[DEBUG LIME] R²={model.score(masks, predictions, sample_weight=weights):.4f}")

        return {
            'combined': attr,
            'spatial_importance': spatial,
            'temporal_importance': np.mean(attr, axis=0),
            'coefficients': coefficients,
            'n_samples': self.n_samples,
            'unit': self.unit,
            'r2_score': model.score(masks, predictions, sample_weight=weights),
        }

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_samples': 1000,
            'kernel_width': 1.0,
            'unit': 'channel_patch',
        }
