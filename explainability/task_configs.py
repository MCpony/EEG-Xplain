"""
任务配置模块 - 独立于模型和适配器

存放各种EEG任务的通用元信息（与模型无关）。
模型相关参数（n_channels, n_patches, patch_size 等）从模型 yaml 配置获取。
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskConfig:
    """任务通用配置（模型无关）"""
    task_type: str = 'state'  # 'state'=状态识别(spatial), 'event'=事件检测(temporal)
    num_classes: int = 2
    domain: str = ''          # 临床领域: epilepsy, depression, motor_imagery, emotion, sleep, etc.


# ============================================================
# 预定义的任务配置
# ============================================================

TASK_CONFIGS: Dict[str, TaskConfig] = {
    'tuab': TaskConfig(
        task_type='state',
        num_classes=2,
        domain='clinical',
    ),
    'tuev': TaskConfig(
        task_type='event',
        num_classes=6,
        domain='epilepsy',
    ),
    'bciciv2a': TaskConfig(
        task_type='state',
        num_classes=4,
        domain='motor_imagery',
    ),
    'chb': TaskConfig(
        task_type='event',
        num_classes=2,
        domain='epilepsy',
    ),
    'faced': TaskConfig(
        task_type='state',
        num_classes=9,
        domain='emotion',
    ),
    'isruc': TaskConfig(
        task_type='state',
        num_classes=5,
        domain='sleep',
    ),
    'mumtaz': TaskConfig(
        task_type='state',
        num_classes=2,
        domain='depression',
    ),
    'physio': TaskConfig(
        task_type='state',
        num_classes=4,
        domain='motor_imagery',
    ),
    'seedv': TaskConfig(
        task_type='state',
        num_classes=5,
        domain='emotion',
    ),
    'seedvig': TaskConfig(
        task_type='state',
        num_classes=2,
        domain='fatigue',
    ),
    'shu': TaskConfig(
        task_type='event',
        num_classes=1,
        domain='motor_imagery',
    ),
    'speech': TaskConfig(
        task_type='state',
        num_classes=5,
        domain='speech_imagery',
    ),
    'stress': TaskConfig(
        task_type='state',
        num_classes=2,
        domain='stress',
    ),
}


def get_task_config(task: str) -> TaskConfig:
    """获取任务配置"""
    task = task.lower()
    if task not in TASK_CONFIGS:
        available = ', '.join(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown task: {task}. Available: {available}")
    return TASK_CONFIGS[task]


def list_tasks() -> Dict[str, str]:
    """列出所有支持的任务"""
    return {name: f"{config.domain} ({config.task_type})"
            for name, config in TASK_CONFIGS.items()}


def register_task(name: str, config: TaskConfig):
    """注册新任务配置"""
    TASK_CONFIGS[name.lower()] = config
