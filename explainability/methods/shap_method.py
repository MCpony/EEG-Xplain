"""
SHAP (SHapley Additive exPlanations) 实现
"""
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from ..base import ExplainabilityMethod, ExplainabilityRegistry


@ExplainabilityRegistry.register('shap')
class SHAPMethod(ExplainabilityMethod):
    """
    SHAP (SHapley Additive exPlanations)

    基于Shapley值的归因方法
    """

    name = "shap"
    description = "SHAP - 基于博弈论Shapley值的归因，提供一致的特征重要性"

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda',
                 n_background: int = 50, method: str = 'kernel'):
        """
        Args:
            model_adapter: 模型适配器
            device: 计算设备
            n_background: 背景样本数量
            method: SHAP方法，目前只支持 'kernel'（真正模型无关的 KernelSHAP）
                    'gradient' 和 'deep' 对 Transformer 不兼容，已废弃
        """
        super().__init__(model_adapter, device)
        self.n_background = n_background
        self.method = method

        # 检查shap库
        self._use_shap = self._check_shap()

    def _check_shap(self) -> bool:
        """检查是否可用shap库"""
        try:
            import shap
            return True
        except ImportError:
            print("警告: shap库未安装，将使用简化版SHAP")
            return False

    def _generate_background(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """生成背景样本"""
        # 使用高斯噪声作为背景
        background = torch.randn(self.n_background, *input_tensor.shape[1:],
                                 device=self.device) * input_tensor.std()
        return background

    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                background: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, np.ndarray]:
        """
        计算SHAP值

        Args:
            input_tensor: 输入张量
            target: 目标类别
            background: 背景数据（可选）

        Returns:
            归因结果字典
        """
        input_tensor = input_tensor.to(self.device)

        if background is None:
            background = self._generate_background(input_tensor)

        if self._use_shap:
            return self._explain_shap(input_tensor, background, target)
        else:
            return self._explain_simple(input_tensor, background, target)

    def _explain_shap(self, input_tensor: torch.Tensor, background: torch.Tensor,
                      target: Optional[int]) -> Dict[str, np.ndarray]:
        """使用shap库计算SHAP值（patch-level KernelSHAP）。

        以 (C, N_patches) 为特征空间而非逐元素展开，原因：
        1. 避免 O(C × N × P) 维度导致的 OOM
        2. patch-level Shapley 值语义更清晰（整段时间窗的贡献）
        3. 与其他方法输出的 (C, N_patches) 粒度对齐
        每个超特征被 mask=0 时用背景均值替换整个 patch 向量。
        """
        import shap

        # input_tensor: (1, C, N, P) 或 (1, C, T)
        original_shape = input_tensor.shape[1:]  # (C, N, P) 或 (C, T)

        if len(original_shape) == 3:
            C, N, P = original_shape
            # 背景均值：每个 (c, n) patch 的均值向量，shape (C, N, P)
            bg_mean = background.mean(dim=0)  # (C, N, P)

            def model_fn(mask2d):
                # mask2d: (n_samples, C*N)，值为 0 或 1
                n_samples = mask2d.shape[0]
                x = bg_mean.unsqueeze(0).expand(n_samples, -1, -1, -1).clone()  # (n_s, C, N, P)
                inp = input_tensor.expand(n_samples, -1, -1, -1)  # (n_s, C, N, P)
                mask = torch.from_numpy(mask2d).float().to(self.device)  # (n_s, C*N)
                mask = mask.reshape(n_samples, C, N, 1)  # 广播到 P 维
                x = x * (1 - mask) + inp * mask
                with torch.no_grad():
                    out = self.model_adapter.forward(x)
                return out.cpu().numpy()

            # 背景：每个背景样本的 patch-level mask（全1，因为我们用背景均值作为0-feature）
            # KernelSHAP 需要一个"reference"输入，这里用全0 mask（即全背景均值）
            background_ref = np.zeros((1, C * N))  # 代表"所有 patch 都用背景均值"
            input_mask = np.ones((1, C * N))        # 代表"所有 patch 都用真实值"

        else:
            # 2D 输入 (C, T)：按 patch_size 分组
            C, T = original_shape
            info = self.model_adapter.get_model_info()
            patch_size = info.get('patch_size', None)
            if patch_size is None or patch_size > T:
                print("  [SHAP] 无法确定 patch_size，退回简化版。")
                return self._explain_simple(input_tensor, background, target)
            N = T // patch_size
            P = patch_size
            T_used = N * P
            bg_mean = background.mean(dim=0)  # (C, T)

            def model_fn(mask2d):
                n_samples = mask2d.shape[0]
                x = bg_mean.unsqueeze(0).expand(n_samples, -1, -1).clone()  # (n_s, C, T)
                inp = input_tensor.expand(n_samples, -1, -1)
                mask = torch.from_numpy(mask2d).float().to(self.device)  # (n_s, C*N)
                mask = mask.reshape(n_samples, C, N, 1).expand(-1, -1, -1, P)
                mask = mask.reshape(n_samples, C, T_used)
                # 只对前 T_used 个时间点做 mask，尾部保留背景均值
                x[:, :, :T_used] = x[:, :, :T_used] * (1 - mask) + inp[:, :, :T_used] * mask
                with torch.no_grad():
                    out = self.model_adapter.forward(x)
                return out.cpu().numpy()

            background_ref = np.zeros((1, C * N))
            input_mask = np.ones((1, C * N))

        # 先做一次前向推理，确定 pred_class 和是否二分类
        with torch.no_grad():
            _out = self.model_adapter.forward(input_tensor)
        is_binary = (_out.numel() == 1 or (_out.dim() == 2 and _out.shape[1] == 1))
        if target is None:
            if is_binary:
                target = int((_out.squeeze() > 0).item())
            else:
                target = int(_out.argmax(dim=-1).item())

        # Wrap model_fn to return only target class logit (scalar per sample)
        # This ensures KernelSHAP returns (1, C*N) instead of (n_classes, 1, C*N)
        _raw_model_fn = model_fn

        def target_model_fn(mask2d):
            out = _raw_model_fn(mask2d)  # (n_samples, n_classes) or (n_samples, 1)
            out = np.array(out)
            if out.ndim == 1:
                return out
            if is_binary:
                return out.reshape(-1)
            return out[:, target]

        try:
            if self.method in ('gradient', 'deep'):
                print(f"  [SHAP] '{self.method}' 变体对 Transformer 不兼容，自动切换为 kernel。")
            explainer = shap.KernelExplainer(target_model_fn, background_ref)
            shap_values = explainer.shap_values(input_mask, nsamples=min(self.n_background * 10, 500))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().numpy()

        except Exception as e:
            print(f"SHAP计算失败: {e}, 使用简化版")
            return self._explain_simple(input_tensor, background, target)

        # shap_values: (1, C*N) → reshape 为 (C, N)
        shap_values = np.array(shap_values).flatten()
        if shap_values.size != C * N:
            print(f"  [SHAP] Unexpected shap_values size {shap_values.size}, expected {C*N}. Falling back.")
            return self._explain_simple(input_tensor, background, target)
        attr = shap_values.reshape(C, N)

        # 二分类 class 0：logit 越低越支持 class 0，需要取反，与其他方法保持一致
        if is_binary and target == 0:
            attr = -attr

        abs_max = np.abs(attr).max()
        if abs_max > 1e-8:
            attr = attr / abs_max

        return {
            'combined': attr,
            'spatial_importance': np.mean(attr, axis=-1),
            'temporal_importance': np.mean(attr, axis=0),
            'method': 'kernel_patch_level',
            'n_background': self.n_background,
        }

    def _explain_simple(self, input_tensor: torch.Tensor, background: torch.Tensor,
                        target: Optional[int]) -> Dict[str, np.ndarray]:
        """简化版SHAP（基于期望梯度）"""
        use_double = getattr(self.model_adapter, 'supports_double', True)
        if use_double:
            input_tensor = input_tensor.double()
            background = background.double()
            self.model_adapter.model.double()
        else:
            input_tensor = input_tensor.float()
            background = background.float()
            self.model_adapter.model.float()

        try:
            # 使用期望梯度近似SHAP
            n_samples = min(self.n_background, 20)  # 限制样本数以加速
            gradients_list = []

            for i in range(n_samples):
                # 在输入和背景之间插值
                alpha = torch.rand(1, device=self.device)
                interpolated = background[i:i+1] + alpha * (input_tensor - background[i:i+1])
                interpolated.requires_grad = True

                # 前向传播
                output = self.model_adapter.forward(interpolated)

                # 确定目标
                if target is None:
                    if output.numel() == 1:
                        target_output = output
                    else:
                        target_output = output.max()
                else:
                    if output.numel() == 1:
                        raw = output.reshape(-1)[0]
                        target_output = -raw if target == 0 else raw
                    elif output.dim() > 1:
                        target_output = output[0, target]
                    else:
                        target_output = output[target]

                # 反向传播
                self.model_adapter.model.zero_grad()
                target_output.backward(retain_graph=True)

                if interpolated.grad is not None:
                    grad = interpolated.grad.clone()
                    grad_norm = torch.norm(grad)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        pass
                    else:
                        if grad_norm > 1e4:
                            grad = grad * (1e4 / grad_norm)
                        gradients_list.append(grad)

                del interpolated, output, target_output

            # 平均梯度
            if not gradients_list:
                print(f"Warning [SHAP]: All gradient steps had NaN/Inf on this device. "
                      f"Results zeroed out. Possible cause: gradient overflow.")
            elif len(gradients_list) < n_samples:
                print(f"Warning [SHAP]: {n_samples - len(gradients_list)}/{n_samples} gradient steps "
                      f"had NaN/Inf and were skipped.")
            if gradients_list:
                avg_grad = torch.stack(gradients_list).mean(dim=0)
            else:
                avg_grad = torch.zeros_like(input_tensor)

            # 计算SHAP近似
            shap_values = avg_grad * (input_tensor - background.mean(dim=0))

            # 转换为numpy
            attr = shap_values.squeeze().cpu().detach().numpy()

            # 处理形状（简化版已经是多维，不需要还原 flatten）
            attr = self._process_shap_values(attr, original_shape=None)

            return {
                'combined': attr,
                'spatial_importance': np.mean(attr, axis=-1),
                'temporal_importance': np.mean(attr, axis=0),
                'method': 'expected_gradients',
                'n_background': n_samples,
                'note': 'simplified_version',
            }

        finally:
            self.model_adapter.model.float()

    def _process_shap_values(self, shap_values: np.ndarray, original_shape=None) -> np.ndarray:
        """处理SHAP值形状，保留符号"""
        attr = np.squeeze(shap_values)

        # KernelSHAP 返回 flatten 的 1D，需要还原
        if attr.ndim == 1 and original_shape is not None:
            attr = attr.reshape(original_shape)

        if attr.ndim == 3:
            # (channels, patches, features) -> (channels, patches)
            attr = attr.sum(axis=-1)
        elif attr.ndim == 4:
            # (batch, channels, patches, features) -> (channels, patches)
            attr = attr[0].sum(axis=-1)

        # 保留符号，以绝对值最大值归一化
        abs_max = np.abs(attr).max()
        if abs_max > 1e-8:
            attr = attr / abs_max
        return attr

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_background': 50,
            'method': 'kernel',
        }
