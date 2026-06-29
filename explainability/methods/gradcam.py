"""
GradCAM及其变体实现
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from ..base import ExplainabilityMethod, ExplainabilityRegistry
 

@ExplainabilityRegistry.register('gradcam')
class GradCAMMethod(ExplainabilityMethod):
    """
    GradCAM (Gradient-weighted Class Activation Mapping)

    支持多种变体: GradCAM, GradCAM++, XGradCAM, LayerCAM, FullGrad
    """

    name = "gradcam"
    description = "GradCAM - 基于梯度的类激活映射，可视化模型关注区域"

    # 支持的GradCAM变体
    VARIANTS = ['gradcam', 'gradcam++', 'xgradcam', 'layercam', 'fullgrad']

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda',
                 variant: str = 'gradcam', target_layer: Optional[nn.Module] = None):
        """
        Args:
            model_adapter: 模型适配器
            device: 计算设备
            variant: GradCAM变体名称
            target_layer: 目标层（可选，默认使用适配器提供的）
        """
        super().__init__(model_adapter, device)
        self.variant = variant.lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"不支持的GradCAM变体: {variant}. 可用: {self.VARIANTS}")

        self.target_layer = target_layer or model_adapter.get_target_layer()
        self.reshape_transform = model_adapter.get_reshape_transform()

        # Hook存储
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        计算GradCAM归因

        Args:
            input_tensor: 输入张量
            target: 目标类别索引（None表示使用预测类别）

        Returns:
            归因结果字典
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # 前向传播
        self.model_adapter.model.zero_grad()
        output = self.model_adapter.forward(input_tensor)

        # 确定目标类别
        if target is None:
            if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
                # 二分类 sigmoid 输出
                target = (output.squeeze() > 0).long().item()
            else:
                # 多分类 softmax 输出
                target = output.argmax(dim=-1).item()

        print(f"[DEBUG] output shape: {output.shape}, target class: {target}")

        # 创建目标梯度 - 关键修复：二分类时需要处理梯度方向
        if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
            # 二分类情况：输出是单个 logit
            target_output = output.squeeze()
            # 如果目标是类别0，需要取负号，让梯度指向"增加类别0概率"的方向
            if target == 0:
                target_output = -target_output
                print(f"[DEBUG] Binary classification: target=0, negating output for correct gradient direction")
        else:
            # 多分类情况
            target_output = output[0, target] if output.dim() == 2 else output

        # 反向传播
        target_output.backward(retain_graph=True)

        # 获取激活和梯度
        activations = self.activations
        gradients = self.gradients

        # 应用reshape变换（如果有）
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        # 计算CAM
        cam = self._compute_cam(activations, gradients)

        # 后处理
        cam = self._postprocess_cam(cam, input_tensor.shape)

        # 计算通道和时间重要性
        spatial_importance = np.mean(cam, axis=-1)
        temporal_importance = np.mean(cam, axis=0)

        return {
            'combined': cam,
            'spatial_importance': spatial_importance,
            'temporal_importance': temporal_importance,
            'raw_activations': activations.cpu().numpy(),
            'raw_gradients': gradients.cpu().numpy(),
            'variant': self.variant,
            'target_class': target,
        }

    def _compute_cam(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """根据变体计算CAM"""
        if self.variant == 'gradcam':
            return self._gradcam(activations, gradients)
        elif self.variant == 'gradcam++':
            return self._gradcam_pp(activations, gradients)
        elif self.variant == 'xgradcam':
            return self._xgradcam(activations, gradients)
        elif self.variant == 'layercam':
            return self._layercam(activations, gradients)
        elif self.variant == 'fullgrad':
            return self._fullgrad(activations, gradients)
        else:
            return self._gradcam(activations, gradients)

    def _gradcam(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """标准GradCAM - 适配EEG模型"""
        # 调试信息
        print(f"[DEBUG] activations shape: {activations.shape}")
        print(f"[DEBUG] gradients shape: {gradients.shape}")
        print(f"[DEBUG] activations range: [{activations.min().item():.4f}, {activations.max().item():.4f}]")
        print(f"[DEBUG] gradients range: [{gradients.min().item():.4f}, {gradients.max().item():.4f}]")

       
        if activations.dim() == 4:
            # (batch, features, channels, patches) 格式
            # 在空间维度 (channels, patches) 上求平均，得到每个特征的全局重要性权重
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (batch, 200, 1, 1)
            print(f"[DEBUG] weights shape: {weights.shape}")
            print(f"[DEBUG] weights range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")

            # 200个特征图，每个乘以对应权重，然后在特征维度求和
            cam = torch.sum(weights * activations, dim=1)  # (batch, 20, 5)
            print(f"[DEBUG] cam (before relu) shape: {cam.shape}")
            print(f"[DEBUG] cam (before relu) range: [{cam.min().item():.4f}, {cam.max().item():.4f}]")
        else:
            # 回退到标准2D处理
            weights = torch.mean(gradients, dim=(-2, -1), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)

        # ReLU - 只保留正向贡献（修复梯度方向后应该有足够的正值）
        cam = torch.relu(cam)
        print(f"[DEBUG] cam (after relu) range: [{cam.min().item():.4f}, {cam.max().item():.4f}]")

        result = cam.squeeze().cpu().numpy()
        print(f"[DEBUG] final cam shape: {result.shape}")
        print(f"[DEBUG] final cam unique values count: {len(np.unique(result))}")

        return result

    def _gradcam_pp(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """GradCAM++ - 适配EEG模型"""
        if activations.dim() == 4:
            # (batch, features, channels, patches) 格式
            grad_2 = gradients ** 2
            grad_3 = grad_2 * gradients
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-8)
            alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

            weights = torch.sum(alpha * torch.relu(gradients), dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)
        else:
            grad_2 = gradients ** 2
            grad_3 = grad_2 * gradients
            sum_activations = torch.sum(activations, dim=(-2, -1), keepdim=True)
            alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-8)
            alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

            weights = torch.sum(alpha * torch.relu(gradients), dim=(-2, -1), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)

        cam = torch.relu(cam)
        return cam.squeeze().cpu().numpy()

    def _xgradcam(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """XGradCAM - 适配EEG模型"""
        if activations.dim() == 4:
            # (batch, features, channels, patches) 格式
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True) + 1e-8
            weights = torch.sum(gradients * activations, dim=(2, 3), keepdim=True) / sum_activations
            cam = torch.sum(weights * activations, dim=1)
        else:
            sum_activations = torch.sum(activations, dim=(-2, -1), keepdim=True) + 1e-8
            weights = torch.sum(gradients * activations, dim=(-2, -1), keepdim=True) / sum_activations
            cam = torch.sum(weights * activations, dim=1)

        cam = torch.relu(cam)
        return cam.squeeze().cpu().numpy()

    def _layercam(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """LayerCAM - 适配EEG模型"""
        # LayerCAM: 逐元素相乘后ReLU，再在特征维度求和
        cam = torch.relu(gradients * activations)
        cam = torch.sum(cam, dim=1)
        return cam.squeeze().cpu().numpy()

    def _fullgrad(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """FullGrad (简化版) - 适配EEG模型"""
        # FullGrad: 取绝对值后在特征维度求和
        cam = torch.abs(gradients * activations)
        cam = torch.sum(cam, dim=1)
        return cam.squeeze().cpu().numpy()

    def _postprocess_cam(self, cam: np.ndarray, input_shape: tuple) -> np.ndarray:
        """后处理CAM到输入形状"""
        # 归一化到0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 根据需要调整形状
        n_channels = self.model_adapter.get_n_channels()
        n_patches = self.model_adapter.get_n_patches()

        # 如果CAM形状不对，尝试reshape
        if cam.shape != (n_channels, n_patches):
            try:
                if cam.ndim == 1:
                    # 尝试reshape
                    total_size = cam.size
                    if total_size == n_channels * n_patches:
                        cam = cam.reshape(n_channels, n_patches)
                    else:
                        # 插值
                        from scipy.ndimage import zoom
                        zoom_factors = (n_channels / cam.shape[0],) if cam.ndim == 1 else \
                                      (n_channels / cam.shape[0], n_patches / cam.shape[1])
                        cam = zoom(cam, zoom_factors)
                elif cam.ndim == 2 and cam.shape != (n_channels, n_patches):
                    from scipy.ndimage import zoom
                    zoom_factors = (n_channels / cam.shape[0], n_patches / cam.shape[1])
                    cam = zoom(cam, zoom_factors)
            except Exception:
                pass

        return cam

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'variant': 'gradcam',
        }
