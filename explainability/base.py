"""
可解释性方法基类和注册机制
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import numpy as np
import torch
 

class ExplainabilityMethod(ABC):
    """可解释性方法的抽象基类"""

    # 子类需要定义的类属性
    name: str = "base"  # 方法名称
    description: str = "Base explainability method"  # 方法描述

    def __init__(self, model_adapter: 'ModelAdapter', device: str = 'cuda'):
        """
        初始化可解释性方法

        Args:
            model_adapter: 模型适配器，提供统一的模型接口
            device: 计算设备
        """
        self.model_adapter = model_adapter
        self.device = device if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def explain(self, input_tensor: torch.Tensor, target: Optional[int] = None,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        计算可解释性归因

        Args:
            input_tensor: 输入数据张量
            target: 目标类别（用于分类任务）
            **kwargs: 方法特定参数

        Returns:
            Dict包含:
                - 'combined': (channels, time) 主要重要性图
                - 'spatial_importance': (channels,) 通道重要性
                - 'temporal_importance': (time,) 时间重要性
                - 其他方法特定的输出
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """返回方法的配置参数"""
        return {
            'name': self.name,
            'description': self.description,
            'device': self.device,
        }

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """返回方法的默认参数"""
        return {}


class ExplainabilityRegistry:
    """可解释性方法注册表 - 单例模式"""

    _instance = None
    _methods: Dict[str, Type[ExplainabilityMethod]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        装饰器：注册可解释性方法

        Usage:
            @ExplainabilityRegistry.register('gradcam')
            class GradCAMMethod(ExplainabilityMethod):
                ...
        """
        def decorator(method_class: Type[ExplainabilityMethod]):
            method_name = name or method_class.name
            cls._methods[method_name] = method_class
            return method_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[ExplainabilityMethod]:
        """获取已注册的方法类"""
        if name not in cls._methods:
            available = ', '.join(cls._methods.keys())
            raise ValueError(f"未知的可解释性方法: {name}. 可用方法: {available}")
        return cls._methods[name]

    @classmethod
    def list_methods(cls) -> List[str]:
        """列出所有已注册的方法"""
        return list(cls._methods.keys())

    @classmethod
    def get_method_info(cls) -> Dict[str, str]:
        """获取所有方法的信息"""
        return {
            name: method_cls.description
            for name, method_cls in cls._methods.items()
        }

    @classmethod
    def create(cls, name: str, model_adapter: 'ModelAdapter',
               device: str = 'cuda', **kwargs) -> ExplainabilityMethod:
        """
        创建可解释性方法实例

        Args:
            name: 方法名称
            model_adapter: 模型适配器
            device: 计算设备
            **kwargs: 方法特定参数
        """
        method_cls = cls.get(name)
        return method_cls(model_adapter, device=device, **kwargs)


class ExplainabilityResult:
    """可解释性分析结果的封装类"""

    def __init__(self, method_name: str, attributions: Dict[str, np.ndarray],
                 input_tensor: np.ndarray, channel_names: List[str],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            method_name: 使用的方法名称
            attributions: 归因结果字典
            input_tensor: 原始输入数据
            channel_names: 通道名称列表
            metadata: 额外元数据
        """
        self.method_name = method_name
        self.attributions = attributions
        self.input_tensor = input_tensor
        self.channel_names = channel_names
        self.metadata = metadata or {}

    @property
    def combined(self) -> np.ndarray:
        """主要重要性图 (channels, time)"""
        return self.attributions.get('combined', np.array([]))

    @property
    def spatial_importance(self) -> np.ndarray:
        """通道重要性 (channels,)"""
        if 'spatial_importance' in self.attributions:
            return self.attributions['spatial_importance']
        # 从combined计算
        if self.combined.size > 0:
            return np.mean(self.combined, axis=-1)
        return np.array([])

    @property
    def temporal_importance(self) -> np.ndarray:
        """时间重要性 (time,)"""
        if 'temporal_importance' in self.attributions:
            return self.attributions['temporal_importance']
        # 从combined计算
        if self.combined.size > 0:
            return np.mean(np.abs(self.combined), axis=0)
        return np.array([])

    def get_top_channels(self, k: int = 5) -> List[tuple]:
        """获取最重要的k个通道"""
        importance = self.spatial_importance
        if importance.size == 0:
            return []
        indices = np.argsort(importance)[::-1][:k]
        return [(self.channel_names[i], importance[i]) for i in indices]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'method_name': self.method_name,
            'attributions': {k: v.tolist() for k, v in self.attributions.items()},
            'channel_names': self.channel_names,
            'metadata': self.metadata,
        }

    def save(self, path: str):
        """保存结果到文件"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
