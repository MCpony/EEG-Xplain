"""
EEG Explainability Framework
可扩展的EEG模型可解释性分析框架

"""
 
from .base import (
    ExplainabilityMethod,
    ExplainabilityRegistry,
    ExplainabilityResult,
)
from .model_adapter import ModelAdapter, ModelAdapterRegistry
from .visualizer import EEGVisualizer

# 导入方法以触发注册
from .methods import (
    GradCAMMethod,
    SHAPMethod,
    LIMEMethod,
    IntegratedGradientsMethod,
    DeepLiftMethod,
    GradInputMethod,
    OcclusionMethod,
)

# 导入适配器
from .adapters import CBraModAdapter, LaBraMAdapter

__version__ = '1.0.0'

__all__ = [
    # 基类
    'ExplainabilityMethod',
    'ExplainabilityRegistry',
    'ExplainabilityResult',
    'ModelAdapter',
    'ModelAdapterRegistry',
    'EEGVisualizer',
    # 具体方法
    'GradCAMMethod',
    'SHAPMethod',
    'LIMEMethod',
    'IntegratedGradientsMethod',
    'OcclusionMethod',
    # 适配器
    'CBraModAdapter',
    'LaBraMAdapter',
]


def list_methods():
    """列出所有可用的可解释性方法"""
    return ExplainabilityRegistry.list_methods()


def get_method_info():
    """获取所有方法的描述信息"""
    return ExplainabilityRegistry.get_method_info()


def list_adapters():
    """列出所有可用的模型适配器"""
    return ModelAdapterRegistry.list_adapters()
