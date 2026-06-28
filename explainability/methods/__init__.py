"""
可解释性方法模块

已废弃（代码保留，不再调用）：
  - DeepLiftMethod: Transformer 不兼容，由 GradientShap 替代
  - GradInputMethod: 被 IG / GradientShap 覆盖
"""
from .gradcam import GradCAMMethod
from .shap_method import SHAPMethod
from .lime_method import LIMEMethod
from .integrated_gradients import IntegratedGradientsMethod
from .gradient_shap import GradientShapMethod
from .grad_input import GradInputMethod   # 保留代码，不在 supported_methods 中
from .deeplift import DeepLiftMethod       # 保留代码，不在 supported_methods 中
from .occlusion import OcclusionMethod

__all__ = [
    'GradCAMMethod',
    'SHAPMethod',
    'LIMEMethod',
    'IntegratedGradientsMethod',
    'GradientShapMethod',
    'OcclusionMethod',
]
