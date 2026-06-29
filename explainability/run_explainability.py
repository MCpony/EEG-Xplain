#!/usr/bin/env python
"""
通用可解释性分析框架 - 支持多种模型架构
Generic Explainability Framework - Supports Multiple Model Architectures

Usage:
    python run_explainability.py --model-type cbramod --task tuab --checkpoint model.pth --data sample.npy --method gradcam
    python run_explainability.py --model-type eegmamba --task stress --checkpoint model.pth --data sample.npy --method lime
"""

import argparse
import numpy as np
import torch
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from explainability.task_configs import TASK_CONFIGS
from explainability import ExplainabilityRegistry, EEGVisualizer, get_method_info, ModelAdapterRegistry
from explainability.spectral_attribution import compute_patch_band_correlation
from explainability.spectral_band_attribution import (
    BAND_METHOD_MAP,
    DEFAULT_BANDS,
    plot_band_topomap,
)


# ===================== 输出目录辅助函数 =====================

def _build_sample_output_dir(
    base_dir: str,
    task: str,
    model_type: str,
    checkpoint_path: Optional[str],
    method: str,
    index: int,
) -> str:
    """构建单样本分析的标准化输出目录。"""
    ckpt_stem = Path(checkpoint_path).stem if checkpoint_path else 'no_ckpt'
    return os.path.join(base_dir, 'sample', task, f'{model_type}_{ckpt_stem}', f'idx_{index}', method)


def _write_run_info(output_dir: str, **kwargs):
    """在 output_dir 下写 run_info.txt，记录本次分析的关键元信息。"""
    from datetime import datetime
    os.makedirs(output_dir, exist_ok=True)
    lines = [f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    topk_channels = kwargs.pop('_topk_channels', None)
    for k, v in kwargs.items():
        lines.append(f"{k:<16} : {v}")
    if topk_channels:
        lines.append("")
        lines.append("Top-K Channel Attribution Scores:")
        for name, score in topk_channels:
            sign = '+' if score >= 0 else ''
            lines.append(f"  {name:<12} : {sign}{score:.4f}")
    path = os.path.join(output_dir, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Saved] {path}")


def compute_spatial_importance(combined: np.ndarray, mode: str = 'relu_mean') -> np.ndarray:
    """
    从 combined (C, N) 归因图计算通道重要性 (C,)。

    Args:
        combined: (channels, patches) 归因矩阵
        mode:
            'relu_mean'  - 只统计正 patch 的均值（默认）
            'signed_mean'- 正负 patch 均值（有方向，可能抵消）
            'abs_mean'   - 绝对值均值（反映总活跃度）
    """
    if mode == 'relu_mean':
        return np.mean(np.maximum(combined, 0), axis=-1)
    elif mode == 'abs_mean':
        return np.mean(np.abs(combined), axis=-1)
    else:  # signed_mean
        return np.mean(combined, axis=-1)


def _save_topomap_json(
    output_dir: str,
    spatial_importance: np.ndarray,
    channel_names: list,
    meta: dict,
):
    """保存单样本 topomap 数值为 JSON。"""
    topomap_data = {
        "meta": meta,
        "top_k_channels": [
            channel_names[i]
            for i in np.argsort(spatial_importance)[::-1][:10]
        ],
        "channel_importance": {
            ch: round(float(v), 6)
            for ch, v in zip(channel_names, spatial_importance)
        },
    }
    json_path = os.path.join(output_dir, 'topomap_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(topomap_data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {json_path}")


# ===================== Model Creation Registry =====================

class ModelFactory:
    """工厂类，负责创建不同类型的模型"""

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, model_type: str):
        """注册模型创建函数"""
        def decorator(func):
            cls._registry[model_type] = func
            return func
        return decorator

    @classmethod
    def create_model(cls, model_type: str, config: dict, checkpoint_path: Optional[str] = None, **kwargs):
        """创建模型实例

        Args:
            model_type: 模型类型 (cbramod, eegmamba, eegpt, etc.)
            config: 配置字典，包含 n_channels, n_patches, num_classes, patch_size
            checkpoint_path: 权重文件路径
            **kwargs: 其他模型特定参数

        Returns:
            model: 创建的模型实例
        """
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._registry.keys())}")

        return cls._registry[model_type](config, checkpoint_path, **kwargs)

    @classmethod
    def list_models(cls):
        """列出所有已注册的模型类型"""
        return list(cls._registry.keys())


# ===================== 模型注册 =====================

@ModelFactory.register('cbramod')
def create_cbramod_model(config: dict, checkpoint_path: Optional[str] = None, **kwargs):
    """
    ���建 CBraMod 模型（使用 CBraModLoader）

    Args:
        config: 配置字典，必须包含 n_channels, n_patches, num_classes, patch_size
        checkpoint_path: 模型权重路径
        **kwargs: 可选参数
            - classifier_type: str (default: 'all_patch_reps')
            - d_model: int (default: 200)
            - dropout: float (default: 0.1)
            - freeze_backbone: bool (default: False)

    Returns:
        model: 创建的模型实例
    """
    from load_model.load_finetune_cbramod import CBraModLoader

    # Validate required config parameters

    if not checkpoint_path:
        raise ValueError("CBraMod requires --checkpoint parameter")

    # Debug: print config
    print(f"[DEBUG] Config received in create_cbramod_model: {config}")

    # Use CBraModLoader with config override
    try :
        model = CBraModLoader.load(
        checkpoint_path=checkpoint_path,
        device='cpu',  # Load to CPU first, will be moved to device later
        n_channels=config['n_channels'],
        n_patches=config['n_patches'],
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        classifier_type=config['classifier_type'],
        dropout=config['dropout'],
        freeze_backbone=config['freeze_backbone']
    )
    except Exception as e:
        raise ValueError(f"Failed to load CBraMod model: {e}")

    return model

@ModelFactory.register('labram')
def create_labram_model(config: dict, checkpoint_path: Optional[str] = None, **_):
    """
    创建 LaBraM 模型（使用 LaBraMLoader）

    Args:
        config: 配置字典，必须包含 num_classes
        checkpoint_path: 模型权重路径
        **kwargs: 可选参数
            - model_name: str (default: 'labram_base_patch200_200')
            - drop_path_rate: float (default: 0.1)
            - drop_rate: float (default: 0.0)
            - attn_drop_rate: float (default: 0.0)
            - qkv_bias: bool (default: False)

    Returns:
        model: 创建的模型实例
    """
    from load_model.load_finetune_labram import LaBraMLoader

    # Validate required config parameters
    if 'num_classes' not in config:
        raise ValueError(
            "LaBraM model requires 'num_classes' in config\n"
            "Please provide it via:\n"
            "  --task <name>  (e.g., --task bciciv2a)\n"
            "  OR\n"
            "  --num-classes <n>"
        )

    if not checkpoint_path:
        raise ValueError("LaBraM requires --checkpoint parameter")

    # Use LaBraMLoader with config
    model = LaBraMLoader.load(
        checkpoint_path=checkpoint_path,
        model_name=config.get('model_name', 'labram_base_patch200_200'),
        device='cpu',  # Load to CPU first, will be moved to device later
        **config,
    )

    return model

@ModelFactory.register('eegmamba')
def create_eegmamba_model(config: dict, checkpoint_path: Optional[str] = None, **kwargs):
    """创建 EEGMamba 模型"""
    from types import SimpleNamespace
    from load_model.load_finetune_eegmamba import EEGMambaLoader

    if not checkpoint_path:
        raise ValueError("EEGMamba requires --checkpoint parameter")

    task = kwargs.get('task', config.get('task'))
    if not task:
        raise ValueError("EEGMamba requires --task parameter")

    param = SimpleNamespace(
        use_pretrained_weights=False,
        foundation_dir=None,
        cuda=0,
        classifier=config.get('classifier_type', 'all_patch_reps'),
        num_of_classes=config['num_classes'],
        dropout=config.get('dropout', 0.5),
    )

    model = EEGMambaLoader.load(
        checkpoint_path=checkpoint_path,
        dataset_name=task,
        param=param,
        device='cpu',
    )

    return model

@ModelFactory.register('eegpt')
def create_eegpt_model(config: dict, checkpoint_path: Optional[str] = None, **_):
    from load_model.load_finetune_eegpt import EEGPTLoader

    if not checkpoint_path:
        raise ValueError("EEGPT requires --checkpoint parameter")

    model_config = config['model']

    # YAML 中的 None 字符串需要转换为 Python None
    if model_config.get('patch_stride') == 'None':
        model_config['patch_stride'] = None

    return EEGPTLoader.load(
        checkpoint_path=checkpoint_path,
        mode=config['mode'],
        device='cpu',
        **model_config
    )


@ModelFactory.register('biot')
def create_biot_model(config: dict, checkpoint_path: Optional[str] = None, **_):
    from load_model.load_finetune_biot import BIOTLoader

    if not checkpoint_path:
        raise ValueError("BIOT requires --checkpoint parameter")

    return BIOTLoader.load(
        checkpoint_path=checkpoint_path,
        device='cpu',
        **config
    )


# ===================== 数据加载 =====================

def load_data(data_path: str, data_key: str = 'eeg') -> tuple:
    """加载单个数据文件，返回 (data, label)"""
    ext = Path(data_path).suffix.lower()
    label = None

    if ext == '.npy':
        data = np.load(data_path)
        # 尝试在同目录找标签文件
        parent = Path(data_path).parent
        for lname in ['y_data.npy', 'labels.npy', 'y.npy', 'label.npy']:
            lpath = parent / lname
            if lpath.exists():
                labels_arr = np.load(str(lpath)).flatten()
                fname = Path(data_path).stem
                # 尝试从文件名提取索引
                import re
                m = re.search(r'(\d+)', fname)
                if m:
                    idx = int(m.group(1))
                    if idx < len(labels_arr):
                        label = int(labels_arr[idx])
                break
    elif ext == '.pt':
        raw = torch.load(data_path)
        if isinstance(raw, torch.Tensor):
            data = raw.numpy()
        elif isinstance(raw, dict):
            # 提取标签
            for lkey in ['label', 'y', 'target']:
                if lkey in raw:
                    lval = raw[lkey]
                    if isinstance(lval, torch.Tensor):
                        label = int(lval.item()) if lval.numel() == 1 else int(lval.flatten()[0].item())
                    else:
                        label = int(lval)
                    break
            # 提取数据
            if 'sample' in raw:
                data = raw['sample']
            elif data_key in raw:
                data = raw[data_key]
            else:
                data = list(raw.values())[0]
            if isinstance(data, torch.Tensor):
                data = data.numpy()
        else:
            data = raw
            if isinstance(data, torch.Tensor):
                data = data.numpy()
    elif ext == '.mat':
        from scipy.io import loadmat
        mat_data = loadmat(data_path)
        data = mat_data[data_key]
        for lkey in ['y_data', 'labels', 'y', 'label']:
            if lkey in mat_data:
                larr = np.asarray(mat_data[lkey]).flatten()
                if larr.size == 1:
                    label = int(larr[0])
                elif data.shape[0] == 1 and larr.size > 0:
                    label = int(larr[0])
                break
    elif ext == '.npz':
        npz_data = np.load(data_path)
        data = npz_data[data_key] if data_key in npz_data else list(npz_data.values())[0]
        for lkey in ['labels', 'y', 'y_data', 'label']:
            if lkey in npz_data:
                larr = npz_data[lkey].flatten()
                if larr.size == 1:
                    label = int(larr[0])
                elif larr.size > 0:
                    label = int(larr[0])
                break
    else:
        print(f"{ext}...")
        raise ValueError(f"Unsupported data format: {ext}")

    return data, label

def load_from_lmdb(lmdb_path: str, index: int) -> tuple:
    """从 LMDB 数据集加载指定索引的样本，返回 (data, label)"""
    import lmdb
    import pickle

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        # 方法1: 尝试预定义的键格式
        key_formats = [
            str(index).encode(),
            f'{index:08d}'.encode(),
            f'sample_{index}'.encode(),
            f'{index}'.encode('ascii'),
        ]

        data = None
        for key in key_formats:
            value = txn.get(key)
            if value is not None:
                data = pickle.loads(value)
                break

        # 方法2: 如果预定义格式都失败，遍历所有键并按索引选择
        if data is None:
            cursor = txn.cursor()
            keys = [k for k, _ in cursor if k != b'__keys__']
            if index >= len(keys):
                env.close()
                raise IndexError(f"Index {index} out of range (total: {len(keys)} samples)")

            key = keys[index]
            value = txn.get(key)
            data = pickle.loads(value)

    env.close()

    # 处理不同的数据格式
    label = None
    if isinstance(data, dict):
        # 提取 label
        label = data.get('label', data.get('y', data.get('target', None)))
        if isinstance(label, (np.ndarray, torch.Tensor)):
            label = int(label.item() if hasattr(label, 'item') else label)
        elif label is not None:
            label = int(label)
        # 提取 data
        data = data.get('sample', data.get('data', data.get('X', data.get('eeg', data))))

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # 确保有 batch 维度
    if isinstance(data, np.ndarray) and data.ndim == 3:
        data = data[np.newaxis, ...]

    # 检查标签是否 1-indexed：扫描少量样本
    if label is not None and label >= 1:
        try:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            sample_labels = []
            with env.begin() as txn:
                for si in range(min(20, index + 10)):
                    if si == index:
                        sample_labels.append(label)
                        continue
                    for fmt in [str(si).encode(), f'{si:08d}'.encode()]:
                        val = txn.get(fmt)
                        if val is not None:
                            sdata = pickle.loads(val)
                            if isinstance(sdata, dict):
                                sl = sdata.get('label', sdata.get('y', sdata.get('target', None)))
                                if sl is not None:
                                    sl = int(sl.item() if hasattr(sl, 'item') else sl)
                                    sample_labels.append(sl)
                            break
            env.close()
            if sample_labels and 0 not in sample_labels and min(sample_labels) >= 1:
                label = label - 1
        except Exception:
            pass

    return data, label

def load_from_dataset(dataset_path: str, index: int, data_format: str = None) -> tuple:
    """
    从数据集加载指定索引的样本，返回 (data, label)

    支持的格式:
    - LMDB (.lmdb, .mdb 或目录含 data.mdb)
    - NPY (.npy) - 多样本数组
    - NPZ (.npz) - 压缩数组
    - PT (.pt) - PyTorch 格式
    - MAT (.mat) - MATLAB 格式，取 x_data 字段
    """
    dataset_path = Path(dataset_path)

    # 自动检测格式
    if data_format is None:
        ext = dataset_path.suffix.lower()
        format_map = {
            '.npy': 'npy',
            '.npz': 'npz',
            '.pt': 'pt',
            '.pth': 'pt',
            '.mat': 'mat',
        }
        if ext in format_map:
            data_format = format_map[ext]
        elif dataset_path.is_dir() or ext in ('.lmdb', '.mdb'):
            # 目录：优先检查是否是真正的 LMDB（含 data.mdb），否则找 .mat 文件
            if dataset_path.is_dir() and (dataset_path / 'data.mdb').exists():
                data_format = 'lmdb'
            elif dataset_path.is_dir():
                mat_files = list(dataset_path.glob('*.mat'))
                if mat_files:
                    dataset_path = mat_files[0]
                    data_format = 'mat'
                else:
                    data_format = 'pt_dir'
            else:
                data_format = 'lmdb'
        else:
            data_format = 'unknown'

    # Loading from dataset (format: {data_format}, index: {index})

    # 根据格式加载
    if data_format == 'pt_dir':
        files = sorted(f for f in dataset_path.iterdir() if f.is_file())
        if index >= len(files):
            raise IndexError(f"Index {index} out of range (total files: {len(files)})")
        f = files[index]
        if f.suffix.lower() == '.pkl':
            import pickle
            with open(f, 'rb') as fp:
                tensor = pickle.load(fp)
        else:
            tensor = torch.load(f, map_location='cpu')
        label = None
        if isinstance(tensor, dict):
            label = tensor.get('label', tensor.get('y', tensor.get('target', None)))
            if isinstance(label, (np.ndarray, torch.Tensor)):
                label = int(label.item() if hasattr(label, 'item') else label)
            elif label is not None:
                label = int(label)
            tensor = tensor.get('data', tensor.get('X', tensor.get('eeg',
                        tensor.get('signal', list(tensor.values())[0]))))
        # 检查标签是否 1-indexed：扫描少量样本
        if label is not None and label >= 1:
            sample_indices = list(range(min(20, len(files))))
            sample_labels = []
            for si in sample_indices:
                try:
                    sf = files[si]
                    if sf.suffix.lower() == '.pkl':
                        import pickle
                        with open(sf, 'rb') as fp:
                            st = pickle.load(fp)
                    else:
                        st = torch.load(sf, map_location='cpu', weights_only=False)
                    if isinstance(st, dict):
                        sl = st.get('label', st.get('y', st.get('target', None)))
                        if sl is not None:
                            sl = int(sl.item() if hasattr(sl, 'item') else sl)
                            sample_labels.append(sl)
                except Exception:
                    pass
            if sample_labels and 0 not in sample_labels and min(sample_labels) >= 1:
                label = label - 1
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        if isinstance(tensor, np.ndarray) and tensor.ndim == 2:
            tensor = tensor[np.newaxis, ...]  # (C, T) -> (1, C, T)
        data = tensor

    elif data_format == 'lmdb':
        data, label = load_from_lmdb(str(dataset_path), index)
        return data, label

    elif data_format == 'npy':
        dataset = np.load(dataset_path)
        if index >= len(dataset):
            raise IndexError(f"Index {index} out of range (dataset size: {len(dataset)})")
        data = dataset[index]
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        # 尝试从同目录下的 label 文件获取标签
        label = None
        label_candidates = ['y_data.npy', 'labels.npy', 'y.npy', 'label.npy']
        for lf in label_candidates:
            label_path = dataset_path.parent / lf
            if label_path.exists():
                labels = np.load(label_path)
                if 0 not in labels and labels.min() >= 1:
                    labels = labels - 1
                if index < len(labels):
                    label = int(labels[index])
                break

    elif data_format == 'npz':
        npz = np.load(dataset_path)
        dataset = npz.get('data', npz.get('X', list(npz.values())[0]))
        if index >= len(dataset):
            raise IndexError(f"Index {index} out of range (dataset size: {len(dataset)})")
        data = dataset[index]
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        # 提取标签
        label = None
        labels_arr = npz.get('labels', npz.get('y', npz.get('y_data', None)))
        if labels_arr is not None:
            if 0 not in labels_arr and labels_arr.min() >= 1:
                labels_arr = labels_arr - 1
            if index < len(labels_arr):
                label = int(labels_arr[index])

    elif data_format == 'pt':
        dataset = torch.load(dataset_path)
        label = None
        if isinstance(dataset, dict):
            # 提取标签
            labels_arr = dataset.get('labels', dataset.get('y', dataset.get('y_data', None)))
            if labels_arr is not None:
                if isinstance(labels_arr, torch.Tensor):
                    labels_arr = labels_arr.numpy()
                if 0 not in labels_arr and labels_arr.min() >= 1:
                    labels_arr = labels_arr - 1
                if index < len(labels_arr):
                    label = int(labels_arr[index])
            dataset = dataset.get('data', dataset.get('X', list(dataset.values())[0]))
        if isinstance(dataset, torch.Tensor):
            dataset = dataset.numpy()
        if index >= len(dataset):
            raise IndexError(f"Index {index} out of range (dataset size: {len(dataset)})")
        data = dataset[index]
        if data.ndim == 3:
            data = data[np.newaxis, ...]

    elif data_format == 'mat':
        import scipy.io
        mat_path = dataset_path
        if mat_path.is_dir():
            mat_files = list(mat_path.glob('*.mat'))
            if not mat_files:
                raise ValueError(f"No .mat files found in {mat_path}")
            mat_path = mat_files[0]
        mat = scipy.io.loadmat(str(mat_path))
        # 优先取 x_data，其次找第一个非私有字段
        if 'x_data' in mat:
            dataset = mat['x_data']
        else:
            keys = [k for k in mat.keys() if not k.startswith('_')]
            if not keys:
                raise ValueError(f"No valid data fields found in {dataset_path}")
            dataset = mat[keys[0]]
            print(f"[MAT] Using field '{keys[0]}'")
        if index >= len(dataset):
            raise IndexError(f"Index {index} out of range (dataset size: {len(dataset)})")
        data = dataset[index]
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data[np.newaxis, ...]
        # 提取标签
        label = None
        labels_arr = mat.get('y_data', mat.get('labels', mat.get('y', None)))
        if labels_arr is not None:
            labels_arr = np.asarray(labels_arr).flatten()
            if 0 not in labels_arr and labels_arr.min() >= 1:
                labels_arr = labels_arr - 1
            if index < len(labels_arr):
                label = int(labels_arr[index])

    else:
        raise ValueError(f"Unsupported dataset format: {data_format}")

    # Loaded sample shape: {data.shape}
    return data, label



def find_tp_sample(
    dataset_path: str,
    tp_class: int,
    tp_index: int,
    model_type: str,
    config: dict,
    task: str,
    checkpoint_path: str = None,
    device: str = 'cuda',
    data_format: str = None,
    conf_threshold: float = 0.7,
    **model_kwargs
) -> tuple:
    """全量推理数据集，筛选高置信度 TP 样本，返回第 tp_index 个。

    流程：
    1. 全量前向推理，收集 (index, pred, prob, label)
    2. 筛选: label==tp_class AND pred==tp_class AND prob > conf_threshold
    3. 按置信度降序排列，取第 tp_index 个

    Returns:
        (data, label, dataset_index) - 样本数据、标签、在数据集中的原始索引
    """
    from explainability.run_population_analysis import collect_predictions

    print(f"\n[TP Search] class={tp_class}, tp_index={tp_index}, conf_threshold={conf_threshold}")

    model = ModelFactory.create_model(
        model_type=model_type,
        config=config,
        checkpoint_path=checkpoint_path,
        task=task,
        **model_kwargs
    )
    model.eval()
    model.to(device)

    adapter = ModelAdapterRegistry.create(
        name=model_type,
        model=model,
        config=config,
        task=task,
        device=device
    )

    _cls_threshold = 0.6394 if model_type == 'biot' else 0.5
    records = collect_predictions(
        adapter=adapter,
        dataset_path=dataset_path,
        device=device,
        data_format=data_format,
        n_samples=-1,
        threshold=_cls_threshold,
    )
    print(f"  Inference done: {len(records)} samples")

    # 自动检测 label/pred 偏移（与 population 一致）
    all_labels = [r.get('label') for r in records if r.get('label') is not None]
    if all_labels:
        pred_set = set(r['pred'] for r in records)
        label_set = set(int(l) for l in all_labels)
        num_classes = config.get('num_classes', None)
        max_valid_pred = max(1, num_classes - 1) if num_classes else (max(pred_set) if pred_set else 0)

        need_shift = False
        if 0 not in label_set and max(label_set) > max_valid_pred:
            need_shift = True
        if not need_shift and len(pred_set & label_set) == 0 and len(pred_set) > 0:
            shifted_label_set = set(l - 1 for l in label_set)
            if pred_set == shifted_label_set or pred_set.issubset(shifted_label_set):
                need_shift = True

        if need_shift:
            print(f"  [Auto-fix] Labels are 1-indexed ({sorted(label_set)}), "
                  f"preds are 0-indexed ({sorted(pred_set)}). Shifting labels by -1.")
            for r in records:
                if r.get('label') is not None:
                    r['label'] = int(r['label']) - 1

    # 诊断: 看看 class 样本的 pred 分布
    class_records = [r for r in records if r.get('label') is not None and int(r['label']) == tp_class]
    from collections import Counter
    pred_dist = Counter(int(r['pred']) for r in class_records)
    print(f"  Class {tp_class}: {len(class_records)} samples, pred distribution: {dict(pred_dist)}")

    # 筛选 TP: label==tp_class AND pred==tp_class
    all_tp_records = [
        r for r in records
        if r.get('label') is not None
        and int(r['label']) == tp_class
        and int(r['pred']) == tp_class
    ]
    all_tp_records.sort(key=lambda r: r['prob'], reverse=True)

    # 再按置信度过滤
    tp_records = [r for r in all_tp_records if r['prob'] >= conf_threshold]

    print(f"  TPs (any conf): {len(all_tp_records)}, TPs with prob >= {conf_threshold}: {len(tp_records)}")
    if all_tp_records and not tp_records:
        top5 = all_tp_records[:5]
        top_probs = [f"{r['prob']:.4f}" for r in top5]
        print(f"  Top TP probs (below threshold): {top_probs}")
        print(f"  Consider lowering --tp-conf (current: {conf_threshold})")

    if tp_index >= len(tp_records):
        raise ValueError(
            f"Only found {len(tp_records)} high-confidence TP samples for class {tp_class} "
            f"(conf > {conf_threshold}, total TPs={len(all_tp_records)}), but tp_index={tp_index} requested. "
            f"Try lowering --tp-conf."
        )

    selected = tp_records[tp_index]
    sel_idx = selected['index']
    print(f"  Selected: dataset_index={sel_idx}, prob={selected['prob']:.4f}")

    # 加载该样本的原始数据
    data, label = load_from_dataset(dataset_path, sel_idx, data_format)
    return data, label, sel_idx


# ===================== Main Explainability Pipeline =====================

def run_explainability(
    model_type: str, #模型名称
    config: dict,
    method: str,     #使用的方法
    data: np.ndarray, #数据
    task: Optional[str] = None, #任务名称（用于获取正确的通道名称）
    checkpoint_path: Optional[str] = None, #模型保存路
    output_dir: str = './explainability_results', #输出文件夹
    device: str = 'cuda',  #设备
    target: Optional[int] = None, #目标
    visualize: bool = True, #是否可视化
    save_formats: List[str] = ['png', 'npy'], #保存格式
    plot_types: List[str] = ['topomap', 'waveform'], #画图类型
    verbose: bool = False, #是否详细输出
    fs: float = 200.0,       # 采样率 (Hz)
    spectral_top_k: int = 5, # Top-K 通道数（用于 band-topomap / patch-band-corr）
    patch_band_corr: bool = False,  # 是否运行 Patch 归因-频段能量相关性分析
    band_topomap: bool = True,     # 单样本频段地形图
    llm: Optional[str] = 'claude',   # LLM 解读: "claude" | "openai" | "deepseek"
    llm_model: Optional[str] = None,
    llm_mode: str = 'picture',      # LLM 模式: 'picture' (多模态) | 'json' (纯文本)
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    spatial_aggregation: str = 'relu_mean',  # 通道重要性聚合方式: relu_mean / signed_mean / abs_mean
    **model_kwargs
):
    """运行可解释性分析
 
    Args:
        model_type: 模型类型 (cbramod, eegmamba, etc.)
        config: 配置字典，包含 n_channels, n_patches, num_classes, patch_size
        method: 可解释性方法 (gradcam, lime, shap, ig)
        data: 输入数据 (numpy array)
        checkpoint_path: 模型权重路径
        output_dir: 输出目录
        device: 设备 (cuda/cpu)
        target: 目标类别索引
        visualize: 是否可视化
        save_formats: 保存格式
        plot_types: 绘图类型
        verbose: 详细输出
        **model_kwargs: 模型特定参数
    """

    #创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 模型创建
    print(f"\n[1/5] Creating {model_type} model...")
    model = ModelFactory.create_model(
        model_type=model_type,
        config=config,
        checkpoint_path=checkpoint_path,
        task=task,
        **model_kwargs
    )
    model.eval()
    model.to(device)

    # 2.创建模型适配器
    print(f"[2/5] Creating adapter for {model_type}...")

    adapter = ModelAdapterRegistry.create(
        name=model_type,
        model=model,
        config=config,
        task=task,
        device=device
    )

    # 显示使用的通道信息
    channel_names = adapter.get_channel_names()
    print(f"  Using {len(channel_names)} channels: {', '.join(channel_names[:10])}" +
          (f", ... (+{len(channel_names)-10} more)" if len(channel_names) > 10 else ""))

    # 3. 准备数据
    print(f"[3/5] Preparing input data (shape: {data.shape})...")
    # 在 preprocess 之前保存原始信号用于频谱归因
    raw_signal_for_spectral = data.squeeze(0) if data.ndim >= 3 else data  # 去掉 batch 维
    if raw_signal_for_spectral.ndim == 3:
        # (n_channels, n_patches, patch_size) -> (n_channels, signal_length)
        raw_signal_for_spectral = raw_signal_for_spectral.reshape(raw_signal_for_spectral.shape[0], -1)
    if isinstance(raw_signal_for_spectral, np.ndarray):
        raw_signal_for_spectral = raw_signal_for_spectral.astype(np.float32).copy()
    data = adapter.prepare_input(data)

    if verbose:
        print(f"  Input tensor shape: {data.shape}")

    # 4.模型输出
    print(f"[4/5] Getting model prediction...")
    with torch.no_grad():
        output = adapter.forward(data)
        if output.numel() == 1:
            pred_prob = torch.sigmoid(output).item()
            threshold = 0.6394 if adapter.model_name == 'biot' else 0.5
            pred_class = int(pred_prob > threshold)
            confidence = pred_prob if pred_class == 1 else 1 - pred_prob
            print(f"  Output(logit): {output.item():.4f}, Prob(sigmoid): {pred_prob:.4f}, Predicted class: {pred_class} (threshold={threshold}), Confidence: {confidence:.4f}")
        else:
            pred_prob = None
            threshold = None
            pred_class = output.argmax(dim=-1).item()
            probs = torch.softmax(output, dim=-1)
            confidence = probs[0, pred_class].item() if output.dim() > 1 else probs[pred_class].item()
            pred_prob = confidence
            print(f"  Predicted class: {pred_class}, Confidence: {confidence:.4f}")

    # 5. 运行解释性方法
    print(f"[5/5] Running {method.upper()} explainability analysis...")

    explainer = ExplainabilityRegistry.create(method, adapter, device=device)
    result = explainer.explain(data, target=target)

    # 统一覆盖 spatial_importance（GradCAM 已全正，三种 mode 结果一致）
    if 'combined' in result:
        result['spatial_importance'] = compute_spatial_importance(
            result['combined'], mode=spatial_aggregation
        )

    # Save results
    if 'npy' in save_formats:
        np.save(os.path.join(output_dir, f'{method}_combined.npy'), result['combined'])
        # spatial/temporal 冗余：spatial 已在 topomap_data.json 中（含通道名），
        # temporal 可从 combined 直接算，不再单独保存
        # np.save(os.path.join(output_dir, f'{method}_spatial.npy'), result['spatial_importance'])
        # np.save(os.path.join(output_dir, f'{method}_temporal.npy'), result['temporal_importance'])
        print(f"  Saved {method}_combined.npy to {output_dir}")

    # if 'json' in save_formats:
    #     metadata = {
    #         'model_type': model_type,
    #         'config': config,
    #         'method': method,
    #         'predicted_class': pred_class,
    #         'channel_names': adapter.get_channel_names(),
    #     }
    #     with open(os.path.join(output_dir, f'{method}_metadata.json'), 'w') as f:
    #         json.dump(metadata, f, indent=2)
    #     print(f"  Saved metadata to {output_dir}")

    # Visualize
    if visualize:
        print(f"\nGenerating visualizations...")
        visualizer = EEGVisualizer()
        channel_names = adapter.get_channel_names()

        # heatmap 和 channel_importance 图已不默认生成（信息与 topomap 重叠）
        # 如需启用，在 --plot-types 中显式指定
        if 'heatmap' in plot_types:
            save_path = os.path.join(output_dir, f'{method}_heatmap.png') if 'png' in save_formats else None
            visualizer.plot_heatmap(
                result['combined'], channel_names,
                title=f'{model_type.upper()} - {method.upper()} Attribution',
                save_path=save_path, show=False
            )

        if 'channel_importance' in plot_types:
            save_path = os.path.join(output_dir, f'{method}_channel_importance.png') if 'png' in save_formats else None
            visualizer.plot_channel_importance(
                result['spatial_importance'], channel_names,
                title=f'{model_type.upper()} - Channel Importance',
                save_path=save_path, show=False
            )

        if 'topomap' in plot_types:
            save_path = os.path.join(output_dir, f'{method}_topomap.png') if 'png' in save_formats else None
            try:
                # GradCAM/Saliency 输出全正，用 0-1 蓝→红表示贡献强度
                # 其他方法有正负，用 signed_mean + (-1,1) 表示方向和强度
                _nonneg_methods = {'gradcam', 'saliency'}
                if method.lower() in _nonneg_methods:
                    topo_values = result['spatial_importance']  # 已全正，abs_mean/relu_mean 均可
                    topo_vlim = (0, 1)
                else:
                    topo_values = np.mean(result['combined'], axis=-1) if 'combined' in result else result['spatial_importance']
                    topo_vlim = (-1, 1)
                visualizer.plot_topomap(
                    topo_values, channel_names,
                    title=f'{model_type.upper()} - Topomap',
                    save_path=save_path, show=False,
                    cmap='RdBu_r', vlim=topo_vlim,
                    metadata={
                        'model': model_type,
                        'method': method,
                        'label': target,
                        'prediction': pred_class,
                        'confidence': confidence if pred_prob is not None else None,
                    }
                )
            except Exception as e:
                print(f"  Warning: Topomap generation failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
            # 保存 topomap 数值 JSON（与画图是否成功无关）
            if 'npy' in save_formats or 'png' in save_formats:
                _save_topomap_json(
                    output_dir=output_dir,
                    spatial_importance=result['spatial_importance'],
                    channel_names=channel_names,
                    meta={
                        "analysis_type": "sample",
                        "model": model_type,
                        "method": method,
                        "task": task,
                        "checkpoint": checkpoint_path,
                        "predicted_class": pred_class,
                        "predicted_prob": round(pred_prob, 4) if pred_prob is not None else None,
                    },
                )

        if 'waveform' in plot_types:
            save_path = os.path.join(output_dir, f'{method}_waveform.png') if 'png' in save_formats else None
            try:
                combined = result['combined']
                # 用原始信号 + patch 级归因画波形
                # raw_signal_for_spectral: (channels, time)
                # combined: (channels, n_patches) — reshape 原始信号匹配 patch 结构
                n_ch, n_patches = combined.shape[0], combined.shape[1]
                total_time = raw_signal_for_spectral.shape[1]
                patch_size = total_time // n_patches
                waveform_3d = raw_signal_for_spectral[:n_ch, :n_patches * patch_size].reshape(n_ch, n_patches, patch_size)
                visualizer.plot_waveform_with_heatmap(
                    waveform_3d,
                    combined,
                    channel_names,
                    title=f'{model_type.upper()} - {method.upper()} Waveform',
                    save_path=save_path, show=False,
                    spatial_importance=result['spatial_importance'],
                )
            except Exception as e:
                print(f"  Warning: Waveform visualization failed: {e}")

    print(f"\n[SUCCESS] Explainability analysis completed!")
    print(f"Results saved to: {output_dir}")

    # Print top channels (spatial topomap)
    channel_names = adapter.get_channel_names()
    importance = result['spatial_importance']
    top_indices = np.argsort(importance)[::-1][:5]
    top_channels = [channel_names[i] for i in top_indices]
    print(f"\n[Spatial Topomap] Top-5 important channels:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {channel_names[idx]}: {importance[idx]:.4f}")

    band_result_matrix = None
    band_names = None

    # Band Topomap (single-sample frequency band attribution)
    if band_topomap:
        print(f"\n[Band Topomap] Running single-sample band attribution...")
        band_method = method if method in BAND_METHOD_MAP else 'occlusion'
        band_func = BAND_METHOD_MAP[band_method]
        freq_bands = DEFAULT_BANDS
        try:
            band_result_matrix = band_func(
                raw_signal=raw_signal_for_spectral,
                adapter=adapter,
                target_class=target if target is not None else pred_class,
                fs=fs,
                freq_bands=freq_bands,
                baseline_mode='zero',
                rng=np.random.default_rng(42),
            )
            band_names = list(freq_bands.keys())
            band_result = {
                'method': band_method,
                'baseline_mode': 'zero',
                'channel_band_importance': band_result_matrix,
                'band_names': band_names,
                'channel_names': channel_names,
                'n_samples': 1,
            }
            band_output_dir = os.path.join(output_dir, 'band_topomap')
            os.makedirs(band_output_dir, exist_ok=True)
            plot_band_topomap(band_result, freq_bands, band_output_dir, save=True, show=False, top_k=spectral_top_k,
                              metadata={
                                  'model': model_type,
                                  'method': band_method,
                                  'label': target,
                                  'prediction': pred_class,
                                  'confidence': confidence if pred_prob is not None else None,
                              })
            print(f"  Band topomap saved to: {band_output_dir}")
            # Print per-band top channels
            for b_idx, bname in enumerate(band_names):
                band_imp = band_result_matrix[:, b_idx]
                band_top_idx = np.argsort(np.abs(band_imp))[::-1][:3]
                top_str = ', '.join(f"{channel_names[i]}({band_imp[i]:.4f})" for i in band_top_idx)
                print(f"  [{bname}] Top-3: {top_str}")
        except Exception as e:
            print(f"  Warning: Band topomap failed: {e}")

    # 6. Patch 归因-频段能量相关性分析
    if patch_band_corr:
        print(f"\n[6] Running Patch-Band Correlation Analysis...")
        corr_output_dir = os.path.join(output_dir, 'patch_band_correlation')
        compute_patch_band_correlation(
            raw_signal=raw_signal_for_spectral,
            combined=result['combined'],
            channel_names=adapter.get_channel_names(),
            patch_size=config.get('patch_size', 200),
            fs=fs,
            spatial_importance=result['spatial_importance'],
            top_k=spectral_top_k,
            output_dir=corr_output_dir,
            save=True,
            show=False,
        )
        print(f"  Patch-band correlation saved to: {corr_output_dir}")

    # 7. LLM 归因解读
    if llm:
        from explainability.llm_interpret import interpret_sample
        interpret_sample(
            config=config,
            channel_names=adapter.get_channel_names(),
            spatial_importance=result['spatial_importance'],
            pred_class=pred_class,
            confidence=confidence,
            method=method,
            model_type=model_type,
            output_dir=output_dir,
            mode=llm_mode,
            llm=llm,
            llm_model=llm_model,
            api_key=api_key,
            api_base=api_base,
            true_label=target,
            task=task,
            combined_attribution=result.get('combined'),
            band_result_matrix=band_result_matrix,
            band_names=band_names,
        )

    return result


def run_multiple_methods(
    model_type: str,
    config: dict,
    methods: List[str],
    data: np.ndarray,
    task: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_dir: str = './explainability_results',
    device: str = 'cuda',
    target: Optional[int] = None,
    save_formats: List[str] = ['png', 'npy'],
    verbose: bool = False,
    **model_kwargs #其他实例化会增加的参数
):
    """运行多个可解释性方法并比较"""

    # Create model once
    print(f"\n[Setup] Creating {model_type} model...")
    model = ModelFactory.create_model(
        model_type=model_type,
        config=config,
        checkpoint_path=checkpoint_path,
        task=task,
        **model_kwargs
    )
    model.eval()
    model.to(device)

    adapter = ModelAdapterRegistry.create(
        name=model_type,
        model=model,
        config=config,
        task=task,
        device=device
    )

    # 显示使用的通道信息
    channel_names = adapter.get_channel_names()
    print(f"  Using {len(channel_names)} channels: {', '.join(channel_names[:10])}" +
          (f", ... (+{len(channel_names)-10} more)" if len(channel_names) > 10 else ""))

    # Prepare data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = adapter.prepare_input(data)

    # Run all methods
    results = {}
    for i, method in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] Running {method.upper()}...")
        try:
            # 重置模型状态（防止前一个方法的影响）
            model.eval()
            for param in model.parameters():
                param.requires_grad = True  # 确保参数可以计算梯度

            # 克隆数据（防止 requires_grad 污染）
            data_clean = data.clone().detach()

            explainer = ExplainabilityRegistry.create(method, adapter, device=device)
            result = results[method] = explainer.explain(data_clean, target=target)

            # Save individual results
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            if 'npy' in save_formats:
                np.save(os.path.join(method_dir, 'combined.npy'), result['combined'])
                # np.save(os.path.join(method_dir, 'spatial.npy'), result['spatial_importance'])
                # np.save(os.path.join(method_dir, 'temporal.npy'), result['temporal_importance'])

        except Exception as e:
            print(f"  Error: {method} failed - {e}")
            results[method] = None

    # Generate comparison visualizations (only when multiple methods)
    valid_results = {k: v for k, v in results.items() if v is not None}

    if len(valid_results) > 1:
        print(f"\n[Comparison] Generating comparison visualizations for {len(valid_results)} methods...")
        visualizer = EEGVisualizer()
        channel_names = adapter.get_channel_names()

        # 3. 脑地形图对比（保留，多方法对比的核心图）
        if 'png' in save_formats:
            print("  - Generating topomap comparison...")
            _nonneg_methods = {'gradcam', 'saliency'}
            topo_dict = {}
            for k, v in valid_results.items():
                if k.lower() in _nonneg_methods:
                    topo_dict[k] = v['spatial_importance']
                else:
                    topo_dict[k] = np.mean(v['combined'], axis=-1) if 'combined' in v else v['spatial_importance']
            save_path = os.path.join(output_dir, 'comparison_topomap.png')
            visualizer.plot_topomap_comparison(
                topo_dict, channel_names,
                title=f'{model_type.upper()} - Topomap',
                save_path=save_path, show=False
            )
            print(f"  ✓ Saved comparison_topomap.png")

    print(f"\n[SUCCESS] All methods completed!")
    print(f"Results saved to: {output_dir}")

    # 频谱归因：对所有方法的结果取平均权重后分析
    return results



def main():
    parser = argparse.ArgumentParser(
        description='Generic EEG Explainability Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model and task selection
    parser.add_argument('--model-type', type=str, help='Model type (cbramod, eegmamba, eegpt, etc.)')
    parser.add_argument('--task', type=str, help='Task name (tuab, bciciv2a, stress, etc.)')
    parser.add_argument('--model-config', type=str, help='Path to YAML config file for model parameters')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')

    # Data and method
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--data', type=str, help='Path to single sample file (.npy, .pt, .mat, .npz)')
    data_group.add_argument('--data-from-dataset', type=str, help='Path to dataset (LMDB, .npy, .npz, .pt) to load sample from')
    parser.add_argument('--index', type=int, default=0, help='Sample index when using --data-from-dataset (default: 0)')
    parser.add_argument('--tp-class', type=int, default=None,
                        help='Find true-positive samples of this class (0-indexed). Use with --tp-index')
    parser.add_argument('--tp-index', type=int, default=0,
                        help='Which TP sample to use (0=first TP, 1=second TP, ...). Sorted by confidence descending. Default: 0')
    parser.add_argument('--tp-conf', type=float, default=0.7,
                        help='Confidence threshold for TP filtering. Default: 0.7')
    parser.add_argument('--data-key', type=str, default='eeg', help='Data key for .mat/.npz files')
    parser.add_argument('--data-format', type=str, choices=['lmdb', 'npy', 'npz', 'pt'],
                        help='Explicitly specify dataset format (auto-detected if not provided)')
    parser.add_argument('--method', type=str, help='Single explainability method')
    parser.add_argument('--methods', type=str, help='Multiple methods (comma-separated)')
    parser.add_argument('--all-methods', action='store_true',
                        help='Run all available methods (excludes aliases)')

    # Output
    parser.add_argument('--output-dir', type=str, default='./explainability_results',
                        help='Output directory for results')
    parser.add_argument('--save-formats', type=str, nargs='+', default=['png'],
                        help='Save formats (png, npy, json). Default: png only')
    parser.add_argument('--plot-types', type=str, nargs='+',
                        default=['topomap', 'waveform'],
                        help='Plot types (heatmap, channel_importance, topomap, waveform)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--target', type=int, default=None,
                        help='Target class index for explanation')

    # Model-specific parameters (mainly for CBraMod)
    parser.add_argument('--classifier-type', type=str, default='all_patch_reps',
                        choices=['avgpooling_patch_reps', 'all_patch_reps_onelayer',
                                'all_patch_reps_twolayer', 'all_patch_reps'],
                        help='Classifier type (for CBraMod)')
   
    # List options
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--list-tasks', action='store_true', help='List available tasks')
    parser.add_argument('--list-methods', action='store_true', help='List available methods')

    # Other
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization')

    # Spectral / band analysis
    parser.add_argument('--spectral-top-k', type=int, default=5,
                        help='Number of top channels for band-topomap / patch-band-corr (default: 5)')
    parser.add_argument('--patch-band-corr', action='store_true',
                        help='Run Patch attribution vs band energy Spearman correlation analysis')
    parser.add_argument('--band-topomap', action='store_true', default=True,
                        help='Run single-sample band topomap (default: on)')
    parser.add_argument('--no-band-topomap', dest='band_topomap', action='store_false',
                        help='Disable single-sample band topomap')
    parser.add_argument('--spatial-aggregation', type=str, default='relu_mean',
                        choices=['relu_mean', 'signed_mean', 'abs_mean'],
                        help='Channel importance aggregation: relu_mean (default, positive patches only), '
                             'signed_mean (net contribution, may cancel), abs_mean (total activity)')

    # Report / LLM
    parser.add_argument('--llm', type=str, default='claude', choices=['claude', 'openai', 'deepseek'],
                        help='LLM provider for scientific interpretation (default: claude)')
    parser.add_argument('--llm-model', type=str, default=None,
                        help='Override LLM model (e.g. claude-opus-4-6 / gpt-4o / deepseek-chat)')
    parser.add_argument('--llm-mode', type=str, default='picture', choices=['picture', 'json'],
                        help='LLM interpretation mode: picture (multimodal) or json (text-only)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for LLM (defaults to ANTHROPIC_API_KEY / OPENAI_API_KEY env var)')
    parser.add_argument('--api-base', type=str, default=None,
                        help='Base URL for LLM API proxy/relay (e.g. https://your-proxy.com/v1)')

    args = parser.parse_args()

    #存在的模型和方法
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 60)
        for model_type in ModelFactory.list_models():
            print(f"  - {model_type}")
        return
 
    if args.list_methods:
        print("\nAvailable explainability methods:")
        print("-" * 60)
        for name, desc in get_method_info().items():
            print(f"  - {name:20s} {desc}")
        return


    #若选定的不存在
    if not args.model_type:
        parser.error("--model-type is required")

    if not args.data and not args.data_from_dataset:
        parser.error("Either --data or --data-from-dataset is required")

    if not args.method and not args.methods and not args.all_methods:
        parser.error("One of --method, --methods, or --all-methods is required")


    #检查可用设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    # Load config from YAML file based on model type and task
    if not args.task:
        raise ValueError("Please specify --task parameter (e.g., --task mumtaz)")

    # Determine YAML config file path based on model type
    config_file = os.path.join(project_root, "configs", f"{args.model_type}.yaml")
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found: {config_file}")

    # Load YAML and extract task config
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    if 'CONFIGS' not in yaml_data or args.task not in yaml_data['CONFIGS']:
        available = list(yaml_data.get('CONFIGS', {}).keys())
        raise ValueError(f"Task '{args.task}' not found in {config_file}. Available: {available}")

    config = yaml_data['CONFIGS'][args.task]
    print(f"\nLoaded config for task '{args.task}' from {config_file}:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data 加载数据
    if args.tp_class is not None:
        # TP sample search mode
        if not args.data_from_dataset:
            raise ValueError("--tp-class requires --data-from-dataset to specify the dataset path")
        data, dataset_label, found_index = find_tp_sample(
            dataset_path=args.data_from_dataset,
            tp_class=args.tp_class,
            tp_index=args.tp_index,
            model_type=args.model_type,
            config=config,
            task=args.task,
            checkpoint_path=args.checkpoint,
            device=args.device,
            data_format=args.data_format,
            conf_threshold=args.tp_conf,
        )
        args.index = found_index
        args.target = args.tp_class
        print(f"  Using TP sample: class={args.tp_class}, tp_index={args.tp_index}, dataset_index={found_index}")
    elif args.data:
        # Load single sample file
        print(f"Loading data from file: {args.data}")
        data, dataset_label = load_data(args.data, args.data_key)
        if dataset_label is not None:
            print(f"  Extracted label from file: {dataset_label}")
    else:
        # Load from dataset
        print(f"Loading from dataset: {args.data_from_dataset}")
        data, dataset_label = load_from_dataset(
            dataset_path=args.data_from_dataset,
            index=args.index,
            data_format=args.data_format
        )
        if dataset_label is not None:
            print(f"  Extracted label from dataset: {dataset_label}")
    print(f"Data shape: {data.shape}")

    # 如果用户没有手动指定 target，使用从数据集提取的标签
    if args.target is None and dataset_label is not None:
        args.target = dataset_label
        print(f"  Using extracted label as target: {args.target}")

    # 检查 target 是否越界（部分数据集标签是 1-indexed）
    num_classes = None
    if isinstance(config.get('model'), dict):
        num_classes = config['model'].get('num_classes')
    elif config.get('num_classes'):
        num_classes = config['num_classes']
    if num_classes and args.target is not None and args.target >= num_classes:
        print(f"  Warning: target={args.target} >= num_classes={num_classes}, converting to 0-indexed (target={args.target - 1})")
        args.target = args.target - 1


    # 选择的方法种类
    if args.all_methods:
        # Run all available methods (excluding aliases)
        all_methods = ExplainabilityRegistry.list_methods()
        methods = [m for m in all_methods]
        method_tag = 'all'
    elif args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
        method_tag = '+'.join(methods)
    else:
        methods = None
        method_tag = args.method

    # 构建结构化输出目录
    index = args.index if args.data_from_dataset else 0
    structured_dir = _build_sample_output_dir(
        base_dir=args.output_dir,
        task=args.task,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        method=method_tag,
        index=index,
    )

    # 公共 run_info kwargs
    _run_info_kwargs = dict(
        Model=args.model_type,
        Checkpoint=args.checkpoint or 'N/A',
        Task=args.task,
        Method=method_tag,
        Dataset=args.data_from_dataset or args.data or 'N/A',
        Index=index,
    )

    
    run_explainability(
        model_type=args.model_type,
        config=config,
        method=args.method,
        data=data,
        task=args.task,
        checkpoint_path=args.checkpoint,
        output_dir=structured_dir,
        device=args.device,
        target=args.target,
        visualize=not args.no_visualize,
        save_formats=args.save_formats,
        plot_types=args.plot_types,
        verbose=args.verbose,
        fs=float(config.get('fs', 200)),
        spectral_top_k=args.spectral_top_k,
        patch_band_corr=args.patch_band_corr,
        band_topomap=args.band_topomap,
        llm=args.llm,
        llm_model=args.llm_model,
        llm_mode=args.llm_mode,
        api_key=args.api_key,
        api_base=args.api_base,
        spatial_aggregation=args.spatial_aggregation,
        classifier_type=config.get('classifier_type', 'all_patch_reps'),
        d_model=config.get('d_model', 200)
    )
    _write_run_info(structured_dir, **_run_info_kwargs)


if __name__ == '__main__':
    main()
