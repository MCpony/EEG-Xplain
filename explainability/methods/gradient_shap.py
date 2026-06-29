"""
GradientShap 实现

基于期望梯度的归因方法，使用随机基线近似 SHAP 值。
比 DeepLift 对 Transformer 架构更兼容，比 IG 更鲁棒（多基线平均）。
"""
import torch
import numpy as np
from typing import Dict, Any, Optional
from ..base import ExplainabilityMethod, ExplainabilityRegistry


@ExplainabilityRegistry.register('gradient_shap')
class GradientShapMethod(ExplainabilityMethod):
    """
    GradientShap

    近似 SHAP 值：对随机采样的基线计算 gradient × (input - baseline)，
    多次平均得到期望梯度归因。
    """

    name = "gradient_shap"
    description = "GradientShap - 期望梯度归因，多随机基线平均，Transformer 兼容"

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda',
                 n_samples: int = 25, baseline_type: str = 'zero',
                 noise_level: float = 0.1):
        """
        Args:
            model_adapter: 模型适配器
            device: 计算设备
            n_samples: 随机基线采样次数（越多越准，越慢）
            baseline_type: 基线类型 ('zero', 'gaussian')
                - 'zero': 全零基线加高斯噪声
                - 'gaussian': 纯高斯噪声基线
            noise_level: 高斯噪声标准差（相对于输入标准差的比例）
        """
        super().__init__(model_adapter, device)
        self.n_samples = n_samples
        self.baseline_type = baseline_type
        self.noise_level = noise_level

    def _sample_baselines(self, input_tensor: torch.Tensor) -> list:
        """采样随机基线"""
        baselines = []
        std = input_tensor.std().item()
        for _ in range(self.n_samples):
            if self.baseline_type == 'gaussian':
                baseline = torch.randn_like(input_tensor) * std * self.noise_level
            else:  # zero + noise
                baseline = torch.zeros_like(input_tensor)
                baseline += torch.randn_like(input_tensor) * std * self.noise_level
            baselines.append(baseline)
        return baselines

    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                **kwargs) -> Dict[str, np.ndarray]:
        input_tensor = input_tensor.to(self.device)
        input_tensor = input_tensor.float()
        self.model_adapter.model.float()

        try:
            # 确定预测类别
            with torch.no_grad():
                output = self.model_adapter.forward(input_tensor)
                if target is None:
                    if output.numel() == 1:
                        pred_class = int((output.squeeze() > 0).item())
                    else:
                        pred_class = output.argmax(dim=-1).item()
                    target = pred_class
                else:
                    pred_class = target

            # 尝试 Captum GradientShap（优先）
            try:
                from captum.attr import GradientShap
                # Captum GradientShap 需要基线分布（stacked tensor）
                baselines_list = self._sample_baselines(input_tensor)
                baseline_dist = torch.cat(baselines_list, dim=0).float()

                gs = GradientShap(self.model_adapter.model)
                # 二分类单 logit 输出：target 必须传 None，否则 Captum 会把 batch 维当成类别维
                # 多分类：传 pred_class 选对应列
                captum_target = None if output.numel() == 1 else pred_class
                attribution = gs.attribute(
                    input_tensor,
                    baselines=baseline_dist,
                    target=captum_target,
                    n_samples=self.n_samples,
                )
                attr = attribution.squeeze().cpu().detach().numpy()
                # Captum 对二分类 class 0 的符号需要取反
                if output.numel() == 1 and pred_class == 0:
                    attr = -attr
                used_captum = True
            except Exception as e:
                print(f"  [GradientShap] Captum 调用失败 ({type(e).__name__}: {e})，使用手动实现。")
                used_captum = False

            if not used_captum:
                attr = self._explain_manual(input_tensor, output, pred_class)

            # 处理形状
            attr = self._process_attribution(attr)

            return {
                'combined': attr,
                'spatial_importance': np.mean(attr, axis=-1),
                'temporal_importance': np.mean(attr, axis=0),
                'baseline_type': self.baseline_type,
                'n_samples': self.n_samples,
            }

        finally:
            pass

    def _explain_manual(self, input_tensor: torch.Tensor,
                        output: torch.Tensor, pred_class: int) -> np.ndarray:
        """手动实现：多随机基线的期望梯度"""
        gradients_list = []
        baselines = self._sample_baselines(input_tensor)

        for baseline in baselines:
            baseline = baseline.float()
            # 在输入和基线之间随机插值
            alpha = torch.rand(1, device=self.device).item()
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            out = self.model_adapter.forward(interpolated)
            self.model_adapter.model.zero_grad()

            if out.numel() == 1:
                raw = out.reshape(-1)[0]
                target_out = -raw if pred_class == 0 else raw
            else:
                target_out = out[0, pred_class] if out.dim() > 1 else out[pred_class]

            target_out.backward()

            if interpolated.grad is not None:
                grad = interpolated.grad.detach()
                grad_norm = torch.norm(grad)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    if grad_norm > 1e4:
                        grad = grad * (1e4 / grad_norm)
                    # gradient × (input - baseline)
                    attr_i = grad * (input_tensor - baseline)
                    gradients_list.append(attr_i)

            del interpolated, out, target_out

        if not gradients_list:
            return np.zeros(input_tensor.squeeze().shape)

        avg_attr = torch.stack(gradients_list).mean(dim=0)
        return avg_attr.squeeze().cpu().detach().numpy()

    def _process_attribution(self, attr: np.ndarray) -> np.ndarray:
        """处理归因形状"""
        if attr.ndim == 3:
            attr = attr.sum(axis=-1)
        elif attr.ndim == 4:
            attr = attr[0].sum(axis=-1)
        elif attr.ndim == 2:
            info = self.model_adapter.get_model_info()
            patch_size = info.get('patch_size', None)
            if patch_size is not None:
                patch_stride = info.get('patch_stride', patch_size)
                C, T = attr.shape
                n_p = (T - patch_size) // patch_stride + 1
                if n_p > 0:
                    attr_patches = np.zeros((C, n_p))
                    for i in range(n_p):
                        attr_patches[:, i] = attr[:, i * patch_stride:i * patch_stride + patch_size].sum(axis=-1)
                    attr = attr_patches
        attr = attr / (np.abs(attr).max() + 1e-8)
        return attr

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_samples': 25,
            'baseline_type': 'zero',
            'noise_level': 0.1,
        }
 