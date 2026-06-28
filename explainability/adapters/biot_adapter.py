"""
BIOT 模型适配器

只负责适配，不负责模型创建和权重加载
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Callable
from ..model_adapter import ModelAdapter, ModelAdapterRegistry
from ..channel_configs import get_channel_names


@ModelAdapterRegistry.register('biot')
class BIOTAdapter(ModelAdapter):
    """
    BIOT 模型适配器

    用于将 BIOT 模型适配到可解释性框架

    Example:
        # 1. 用户自己创建和加载模型
        from explainability.load_model.load_finetune_biot import BIOTLoader
        model = BIOTLoader.load(
            checkpoint_path='finetuned.pth',
            device='cuda',
            **configs['tuab']
        )

        # 2. 用适配器包装
        adapter = BIOTAdapter(model, config=configs['tuab'], task='tuab')

        # 3. 使用可解释性方法
        method = ExplainabilityRegistry.create('gradcam', adapter)
        result = method.explain(input_tensor)
    """

    model_name = "biot"
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
            model: 已创建并加载好权重的BIOTClassifier模型
            config: 模型配置字典（来自biot.yaml）
            task: 任务名称（用于获取正确的通道名称）
            channel_names: 通道名称列表（可选，优先级最高）
            device: 计算设备
        """
        super().__init__(model, device)

        self._n_channels = config['n_channels']
        self._data_channels = config.get('data_channels', self._n_channels)
        self._actual_n_channels = self._data_channels
        self._n_fft = config.get('n_fft', 200)
        self._hop_length = config.get('hop_length', 100)

        num_classes = config.get('num_classes', None)
        if num_classes is None and hasattr(model, 'classifier'):
            num_classes = model.classifier.clshead[-1].out_features
        self.is_binary = (num_classes == 1)
        self.supports_double = False

        if channel_names is not None:
            self._channel_names = channel_names
        else:
            self._channel_names = self._resolve_channel_names(task)

        self._target_layer = self._find_target_layer()

    def _resolve_channel_names(self, task: Optional[str]) -> List[str]:
        if task is None:
            return [f'Ch{i+1}' for i in range(self._data_channels)]

        return get_channel_names('biot', task, self._data_channels)

    def _find_target_layer(self) -> nn.Module:
        """查找目标层用于GradCAM

        BIOT 结构: model.biot.transformer (LinearAttentionTransformer)
        这是 mean pooling 之前、保留完整空间信息的最后一层
        """
        if hasattr(self.model, 'biot'):
            biot = self.model.biot
            if hasattr(biot, 'transformer'):
                return biot.transformer

        for name, module in reversed(list(self.model.named_modules())):
            if 'transformer' in name.lower():
                return module

        return self.model

    def get_target_layer(self) -> nn.Module:
        return self._target_layer

    def get_n_channels(self) -> int:
        return self._actual_n_channels

    def get_n_patches(self) -> Optional[int]:
        """BIOT 的 n_patches 取决于输入时间长度，返回 None 表示动态推算"""
        return None

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def set_channel_names(self, channel_names: List[str]):
        if len(channel_names) != self._n_channels:
            raise ValueError(f"Channel names count ({len(channel_names)}) != n_channels ({self._n_channels})")
        self._channel_names = channel_names

    def get_input_shape(self):
        """BIOT 输入是 3D: (batch, n_channels, time_steps)"""
        return (1, self._n_channels, None)

    def preprocess(self, x) -> torch.Tensor:
        """确保输入为 BIOT 期望的 3D (B, C, T) 格式，并做 95% 分位数归一化

        处理各种可能的输入形状：
        - (C, T) -> (1, C, T)
        - (1, C, T) -> 不变
        - (C, P, F) -> (1, C, P*F)  (patch格式展平)
        - (1, C, P, F) -> (1, C, P*F)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x = x.float()

        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            B, C, P, F = x.shape
            x = x.reshape(B, C, P * F)

        # 95% 分位数归一化（与 BIOT 训练时 __getitem__ 一致）
        q95 = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-8
        x = x / q95

        return x.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if x.dim() == 3:
            self._actual_n_channels = x.shape[1]
        return self.model(x)

    def prepare_input(self, x) -> torch.Tensor:
        """BIOT 期望 (B, C, T) 输入，含 95% 分位数归一化（与训练时 TUABLoader 一致）"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # 95% 分位数归一化
        q95 = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-8
        x = x / q95
        return x.to(self.device)

    def to_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """将 3D 原始时域信号切分为 4D patch 格式，供 LIME/Occlusion 使用

        使用 n_fft 作为 patch_size，与 BIOT 内部 STFT 窗口对齐
        """
        if x.dim() == 4:
            return x
        B, C, T = x.shape
        patch_size = self._n_fft
        n_patches = T // patch_size
        self._original_T = T
        self._tail = x[:, :, n_patches * patch_size:]
        return x[:, :, :n_patches * patch_size].reshape(B, C, n_patches, patch_size)

    def from_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """BIOT forward 期望 3D (B, C, T)，将 4D patch 还原为 3D，拼回尾部"""
        if x.dim() == 3:
            return x
        B, C, N, ps = x.shape
        main = x.reshape(B, C, N * ps)
        if hasattr(self, '_tail') and self._tail is not None and self._tail.shape[-1] > 0:
            tail = self._tail.expand(B, -1, -1).to(main.device)
            return torch.cat([main, tail], dim=-1)
        return main

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'patch_size': self._n_fft,
            'patch_stride': self._hop_length,
            'n_fft': self._n_fft,
            'hop_length': self._hop_length,
            'architecture': 'biot_transformer',
        })
        return info

    def get_reshape_transform(self) -> Optional[Callable]:
        """GradCAM reshape 变换

        Transformer 输出 (B, n_channels * ts, emb_dim)
        需要 reshape 为 (B, emb_dim, n_channels, ts) 供 GradCAM 使用

        n_channels 从实际输入动态获取（模型支持 <= n_channels 的任意通道数）
        """
        adapter = self

        def reshape_fn(tensor):
            if tensor.dim() == 3:
                batch_size, seq_len, emb_dim = tensor.shape
                n_ch = adapter._actual_n_channels
                ts = seq_len // n_ch
                tensor = tensor.view(batch_size, n_ch, ts, emb_dim)
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor

        return reshape_fn
