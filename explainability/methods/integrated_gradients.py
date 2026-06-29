"""
Integrated Gradients 实现
"""
import torch
import numpy as np
from typing import Dict, Any, Optional
from ..base import ExplainabilityMethod, ExplainabilityRegistry


@ExplainabilityRegistry.register('ig')
class IntegratedGradientsMethod(ExplainabilityMethod):
    """
    Integrated Gradients (积分梯度)

    基于路径积分的归因方法，满足完整性公理
    """

    name = "ig"
    description = "积分梯度 - 从基线到输入的梯度积分，理论上严格的归因方法"

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda',
                 n_steps: int = 50, baseline_type: str = 'zero'):
        """
        Args:
            model_adapter: 模型适配器
            device: 计算设备
            n_steps: 积分步数
            baseline_type: 基线类型 ('zero', 'gaussian', 'mean')
        """
        super().__init__(model_adapter, device)
        self.n_steps = n_steps
        self.baseline_type = baseline_type

    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """获取基线"""
        if self.baseline_type == 'zero':
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == 'gaussian':
            gen = torch.Generator(device=input_tensor.device)
            gen.manual_seed(42)
            noise = torch.randn(input_tensor.shape, generator=gen,
                                device=input_tensor.device, dtype=input_tensor.dtype)
            return noise * input_tensor.std()
        elif self.baseline_type == 'mean':
            return torch.ones_like(input_tensor) * input_tensor.mean()
        else:
            return torch.zeros_like(input_tensor)

    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        计算Integrated Gradients

        Args:
            input_tensor: 输入张量
            target: 目标类别

        Returns:
            归因结果字典
        """
        input_tensor = input_tensor.to(self.device)
        baseline = self._get_baseline(input_tensor)

        # 转换为 float64 以防止 CUDA 上的梯度溢出
        # 但某些模型（如Mamba2）的CUDA kernel不支持float64，需跳过
        use_double = getattr(self.model_adapter, 'supports_double', True)
        if use_double:
            input_tensor = input_tensor.double()
            baseline = baseline.double()
            if next(self.model_adapter.model.parameters()).dtype == torch.float32:
                self.model_adapter.model.double()
        else:
            input_tensor = input_tensor.float()
            baseline = baseline.float()

        try:
            # 创建插值路径，所有步合并成一个 batch（数学上等价，速度快 ~50x）
            # eval 模式下 BatchNorm 用 running stats，batch size 不影响结果
            alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
            interp_batch = baseline + alphas.view(-1, *([1] * (input_tensor.dim() - 1))) \
                           * (input_tensor - baseline)
            interp_batch = interp_batch.view(self.n_steps + 1, *input_tensor.shape[1:])
            interp_batch.requires_grad_(True)

            # 一次 forward
            output = self.model_adapter.forward(interp_batch)  # (S+1, num_classes)

            # 对 batch 求和得到 scalar，再 backward
            if target is None:
                if output.shape[-1] == 1 or output.dim() == 1:
                    target_output = output.sum()
                else:
                    target_output = output.max(dim=-1).values.sum()
            else:
                is_sigmoid = getattr(self.model_adapter, 'is_binary', False)
                if is_sigmoid:
                    raw = output.reshape(self.n_steps + 1, -1)[:, 0]
                    target_output = (-raw if target == 0 else raw).sum()
                else:
                    target_output = output[:, target].sum()

            # 一次 backward
            self.model_adapter.model.zero_grad()
            target_output.backward()

            grad = interp_batch.grad  # (S+1, C, T)
            _nan_steps = 0
            if grad is None:
                avg_gradients = torch.zeros_like(input_tensor.squeeze(0))
                _nan_steps = self.n_steps + 1
            else:
                nan_mask = torch.isnan(grad) | torch.isinf(grad)
                _nan_steps = int(nan_mask.flatten(1).any(dim=1).sum().item())
                grad = torch.where(nan_mask, torch.zeros_like(grad), grad)
                avg_gradients = (grad[:-1] + grad[1:]).mean(dim=0) / 2  # (C, T)

            # 积分梯度
            input_1 = input_tensor.squeeze(0)
            baseline_1 = baseline.squeeze(0)
            integrated_grads = (input_1 - baseline_1) * avg_gradients

            if _nan_steps > 0:
                print(f"Warning [IntegratedGradients]: {_nan_steps}/{self.n_steps + 1} steps had NaN/Inf gradients "
                      f"(zeroed out). Results may be less accurate.")

            # 转换为numpy并处理形状
            ig = integrated_grads.detach().cpu().float().numpy()

            # 在特征维度上求和，保留符号
            if ig.ndim == 3:
                # (channels, patches, features) -> (channels, patches)
                ig = ig.sum(axis=-1)
            elif ig.ndim == 4:
                # (batch, channels, patches, features) -> (channels, patches)
                ig = ig[0].sum(axis=-1)
            elif ig.ndim == 2:
                # (channels, T) -> 按 patch 窗口聚合 -> (channels, n_patches)
                info = self.model_adapter.get_model_info()
                patch_size = info.get('patch_size', None)
                if patch_size is not None:
                    patch_stride = info.get('patch_stride', patch_size)
                    C, T = ig.shape
                    n_p = (T - patch_size) // patch_stride + 1
                    ig_patches = np.zeros((C, n_p))
                    for i in range(n_p):
                        ig_patches[:, i] = ig[:, i * patch_stride:i * patch_stride + patch_size].sum(axis=-1)
                    ig = ig_patches

            # 归一化（用绝对值最大值，保留符号）
            ig = ig / (np.abs(ig).max() + 1e-8)

            return {
                'combined': ig,
                'spatial_importance': np.mean(ig, axis=-1),
                'temporal_importance': np.mean(ig, axis=0),
                'n_steps': self.n_steps,
                'baseline_type': self.baseline_type,
            }

        finally:
            pass  # 模型保持 float64，避免重复转换开销；collect_predictions 已在前面完成

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_steps': 50,
            'baseline_type': 'zero',
        }

 