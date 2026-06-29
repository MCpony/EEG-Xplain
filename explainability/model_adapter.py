"""
模型适配器接口 - 隔离模型实现和可解释性方法

适配器的职责：
1. 提供目标层 (GradCAM需要)
2. 提供reshape变换 (Transformer等需要)
3. 提供输入形状信息
4. 提供通道元信息 (用于可视化)
 
适配器不负责：
- 模型创建
- 权重加载
- 任务配置
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np


class ModelAdapter(ABC):
    """
    模型适配器抽象基类

    提供统一的模型接口，使可解释性方法无需关心具体模型实现
    """

    model_name: str = "base"
    supported_methods: List[str] = []

    is_binary: bool = False  # 子类设置：True=sigmoid二分类，False=softmax多分类
    supports_double: bool = True  # 子类设置：False表示模型不支持float64（如Mamba2 CUDA kernels）

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 已创建并加载好权重的PyTorch模型
            device: 计算设备
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.float()
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """模型前向传播"""
        model_dtype = next(self.model.parameters()).dtype
        return self.model(x.to(dtype=model_dtype, device=self.device))

    @abstractmethod 
    def get_target_layer(self) -> nn.Module:
        """获取用于GradCAM等方法的目标层"""
        pass

    @abstractmethod
    def get_n_channels(self) -> int:
        """获取EEG通道数"""
        pass

    @abstractmethod
    def get_n_patches(self) -> int:
        """获取时间patch数"""
        pass

    def get_input_shape(self) -> Tuple[int, ...]:
        """获取模型期望的输入形状"""
        return (1, self.get_n_channels(), self.get_n_patches(), 200)

    def get_channel_names(self) -> List[str]:
        """获取EEG通道名称，默认返回通用名称"""
        return [f'Ch{i+1}' for i in range(self.get_n_channels())]

    def get_reshape_transform(self) -> Optional[Callable]:
        """
        获取GradCAM需要的reshape变换函数

        某些模型（如Transformer）需要特殊的reshape
        默认返回None表示不需要变换
        """
        return None

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """预处理输入数据"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x.to(self.device)

    def prepare_input(self, x) -> torch.Tensor:
        """统一入口：任意格式 → 模型可直接 forward 的 tensor。
        默认流程：ensure batch → to_patch_input(4D) → from_patch_input(模型期望格式)。
        CBraMod 期望 4D，BIOT 期望 3D（from_patch_input 会展平）。
        子类可重写。
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (C, T) -> (1, C, T)
        if x.dim() == 3:
            x = self.to_patch_input(x)  # (B, C, T) -> (B, C, N, ps)
        # 还原为模型期望的格式（CBraMod 保持 4D, BIOT 展平为 3D）
        x = self.from_patch_input(x)
        return x

    def to_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """将输入转为 patch 化的 4D 格式 (B, C, N_patches, patch_size)，供 LIME/Occlusion 使用。
        默认实现：如果已经是 4D 直接返回，3D 时按 n_patches 切分。
        保存尾部未对齐的采样点，from_patch_input 会还原完整 T。
        子类可重写以适配不同模型的 patch 结构。
        """
        if x.dim() == 4:
            return x
        B, C, T = x.shape
        n_patches = self.get_n_patches()
        patch_size = T // n_patches
        used_T = n_patches * patch_size
        self._original_T = T
        self._tail = x[:, :, used_T:]
        return x[:, :, :used_T].reshape(B, C, n_patches, patch_size)

    def from_patch_input(self, x: torch.Tensor) -> torch.Tensor:
        """将 patch 化的 4D 输入还原为模型所需格式。
        默认实现：4D → 3D flatten 时间维，拼回尾部采样点还原原始 T。
        子类可重写以适配不同模型的输入格式。
        """
        if x.dim() == 3:
            return x
        B, C, N, ps = x.shape
        main = x.reshape(B, C, N * ps)
        if hasattr(self, '_tail') and self._tail is not None and self._tail.shape[-1] > 0:
            tail = self._tail.expand(B, -1, -1).to(main.device)
            return torch.cat([main, tail], dim=-1)
        return main

    def postprocess_attribution(self, attribution: np.ndarray) -> np.ndarray:
        """后处理归因结果"""
        return attribution

    def supports_method(self, method_name: str) -> bool:
        """检查是否支持某个可解释性方法"""
        if not self.supported_methods:
            return True
        return method_name in self.supported_methods

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'input_shape': self.get_input_shape(),
            'n_channels': self.get_n_channels(),
            'n_patches': self.get_n_patches(),
            'channel_names': self.get_channel_names(),
            'supported_methods': self.supported_methods,
        }


class ModelAdapterRegistry:
    """模型适配器注册表"""

    _adapters: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """注册模型适配器"""
        def decorator(adapter_cls):
            cls._adapters[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """获取适配器类"""
        if name not in cls._adapters:
            available = ', '.join(cls._adapters.keys())
            raise ValueError(f"Unknown adapter: {name}. Available: {available}")
        return cls._adapters[name]

    @classmethod
    def list_adapters(cls) -> List[str]:
        """列出所有已注册的适配器"""
        return list(cls._adapters.keys())

    @classmethod
    def create(cls, name: str, model: nn.Module, **kwargs) -> ModelAdapter:
        """创建适配器实例"""
        adapter_cls = cls.get(name)
        return adapter_cls(model, **kwargs)
