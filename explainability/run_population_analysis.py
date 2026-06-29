#!/usr/bin/env python
"""
群体归因分析脚本
Population-level Attribution Analysis

对整个数据集进行推理，筛选高置信度样本，
对指定类别进行群体平均频谱归因分析。

Usage:
    python run_population_analysis.py \
        --model-type cbramod \
        --task stress \
        --checkpoint model.pth \
        --data-from-dataset ./data/stress \
        --method gradcam \
        --target-class 1 \
        --conf-threshold 0.85 \
        --n-samples 40 \
        --output-dir ./population_results
"""

import argparse
import numpy as np
import torch
import sys
import os
import time
import yaml
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 复用 run_explainability 里的模型工厂和数据加载
from explainability.run_explainability import (
    ModelFactory,
    load_from_dataset,
    _write_run_info,
    compute_spatial_importance,
)
from explainability import ExplainabilityRegistry, ModelAdapterRegistry
from explainability.spectral_attribution import grand_average_patch_band_correlation

EVENT_TASKS = {'bciciv2a', 'physio', 'shu', 'speech', 'tuev', 'chb'}
STATE_TASKS = {'tuab', 'mumtaz', 'isruc', 'faced', 'seedv', 'seedvig', 'stress'}


def infer_task_type(task: str) -> str:
    t = task.lower()
    if t in EVENT_TASKS:
        return 'event'
    if t in STATE_TASKS:
        return 'state'
    return 'event'


from explainability.faithfulness import spatial_faithfulness, temporal_faithfulness
from explainability.faithfulness import spatial_single_deletion, temporal_single_deletion
from explainability.spectral_band_attribution import (
    grand_band_attribution, compute_band_power_context,
)

# PSD 固定样本数（每个类）
PSD_N_SAMPLES = 100


def _build_population_output_dir(
    base_dir: str,
    task: str,
    model_type: str,
    checkpoint_path: Optional[str],
    method: str,
    target_class: int,
    sample_type: str,
) -> str:
    """构建群体分析的标准化输出目录。"""
    ckpt_stem = Path(checkpoint_path).stem if checkpoint_path else 'no_ckpt'
    return os.path.join(
        base_dir, 'population', task,
        f'{model_type}_{ckpt_stem}', method,
        f'class_{target_class}_{sample_type}',
    )


# ===================== 获取数据集大小 =====================

def get_dataset_size(dataset_path: str, data_format: Optional[str] = None) -> int:
    """自动检测数据集大小，无需用户手动指定"""
    dataset_path = Path(dataset_path)

    # 自动检测格式
    if data_format is None:
        ext = dataset_path.suffix.lower()
        if ext == '.npy':
            data_format = 'npy'
        elif ext == '.npz':
            data_format = 'npz'
        elif ext in ('.pt', '.pth'):
            data_format = 'pt'
        elif ext == '.mat':
            data_format = 'mat'
        elif dataset_path.is_dir() and (dataset_path / 'data.mdb').exists():
            data_format = 'lmdb'
        elif dataset_path.is_dir():
            data_format = 'pt_dir'
        else:
            data_format = 'lmdb'

    if data_format == 'pt_dir':
        files = sorted(f for f in dataset_path.iterdir() if f.is_file())
        return len(files)

    if data_format == 'lmdb':
        import lmdb, pickle
        env = lmdb.open(str(dataset_path), readonly=True, lock=False)
        with env.begin() as txn:
            size = txn.stat()['entries']
            # 过滤掉特殊的 __keys__ 索引条目
            if txn.get(b'__keys__') is not None:
                size -= 1
        env.close()
        return size

    elif data_format == 'npy':
        dataset = np.load(dataset_path, mmap_mode='r')
        return len(dataset)

    elif data_format == 'npz':
        npz = np.load(dataset_path)
        dataset = npz.get('data', npz.get('X', list(npz.values())[0]))
        return len(dataset)

    elif data_format == 'pt':
        dataset = torch.load(dataset_path)
        if isinstance(dataset, dict):
            dataset = list(dataset.values())[0]
        if isinstance(dataset, torch.Tensor):
            return len(dataset)
        return len(dataset)

    elif data_format == 'mat':
        import scipy.io
        mat_path = Path(dataset_path)
        if mat_path.is_dir():
            mat_files = list(mat_path.glob('*.mat'))
            if not mat_files:
                raise ValueError(f"No .mat files found in {dataset_path}")
            mat_path = mat_files[0]
        mat = scipy.io.loadmat(str(mat_path))
        if 'x_data' in mat:
            return len(mat['x_data'])
        keys = [k for k in mat.keys() if not k.startswith('_')]
        return len(mat[keys[0]])

    raise ValueError(f"Cannot determine dataset size for format: {data_format}")


# ===================== Step 1: 全量推理，收集置信度 =====================

def _debug_confidence_stats(
    records,
    target_classes,
    is_binary,
    thresholds=None,
    output_dir=None,
    cls_threshold: float = 0.5,
):
    """
    打印并保存全量推理后各类别的置信度分布统计（TP/FP/各置信度门槛占比）。
    结果同时输出到终端和 output_dir/confidence_stats.txt。
    """
    from collections import Counter

    if thresholds is None:
        _auto_thresholds = True
    else:
        _auto_thresholds = False

    total = len(records)
    has_label = any(r.get("label") is not None for r in records)

    lines = []
    def _p(s=""):
        print(s)
        lines.append(str(s))

    _p("=" * 62)
    _p(f"Confidence Distribution Report  (total={total} samples)")
    _p("=" * 62)

    pred_counter = Counter(r["pred"] for r in records)
    _p("Prediction distribution:")
    for cls, cnt in sorted(pred_counter.items()):
        _p(f"  class {cls}: {cnt:5d} samples  ({cnt/total*100:.1f}%)")

    if has_label:
        label_counter = Counter(int(r["label"]) for r in records if r.get("label") is not None)
        _p("Ground-truth label distribution:")
        for cls, cnt in sorted(label_counter.items()):
            _p(f"  class {cls}: {cnt:5d} samples  ({cnt/total*100:.1f}%)")

    _p()

    for target_class in target_classes:
        _p(f"--- Target class {target_class} ---")

        if _auto_thresholds:
            if target_class == 0:
                boundary = round(1.0 - cls_threshold, 4)
                base = [boundary] if boundary not in [0.5, 0.4, 0.3, 0.2, 0.1] else []
                thresholds = sorted(set(base + [0.5, 0.4, 0.3, 0.2, 0.1]), reverse=True)
            else:
                boundary = round(cls_threshold, 4)
                base = [boundary] if boundary not in [0.5, 0.6, 0.7, 0.8, 0.9] else []
                thresholds = sorted(set(base + [0.5, 0.6, 0.7, 0.8, 0.9]))

        def _conf(r, tc=target_class):
            return r["prob"] if tc != 0 else 1.0 - r["prob"]

        tp_records = [r for r in records if r["pred"] == target_class
                      and (not has_label or r.get("label") is None
                           or int(r["label"]) == target_class)]
        fp_records = [r for r in records if r["pred"] == target_class
                      and has_label and r.get("label") is not None
                      and int(r["label"]) != target_class]

        groups = [("TP", tp_records), ("FP", fp_records)] if is_binary else [("TP", tp_records)]

        for group_name, grp in groups:
            if len(grp) == 0:
                _p(f"  {group_name}: 0 samples")
                continue
            probs = [r["prob"] for r in grp]
            confs = [_conf(r) for r in grp]
            _p(f"  {group_name}  (n={len(grp)},  {len(grp)/total*100:.1f}% of all)")
            _p(f"    prob:  mean={np.mean(probs):.3f}  median={np.median(probs):.3f}"
               f"  min={np.min(probs):.3f}  max={np.max(probs):.3f}")
            _p(f"    {'threshold':<12}  {'n':>8}  {'% of '+group_name:>14}  {'% of all':>10}")
            if target_class == 0:
                for thr in thresholds:
                    n_match = sum(1 for p in probs if p < thr)
                    pct_grp = n_match / len(grp) * 100
                    pct_all = n_match / total * 100
                    thr_str = f"{thr:.4f}" if thr % 0.1 != 0 else f"{thr:.1f}   "
                    _p(f"    < {thr_str}      {n_match:>8d}  {pct_grp:>13.1f}%  {pct_all:>9.1f}%")
            else:
                for thr in thresholds:
                    n_match = sum(1 for c in confs if c >= thr)
                    pct_grp = n_match / len(grp) * 100
                    pct_all = n_match / total * 100
                    thr_str = f"{thr:.4f}" if thr % 0.1 != 0 else f"{thr:.1f}   "
                    _p(f"    >= {thr_str}     {n_match:>8d}  {pct_grp:>13.1f}%  {pct_all:>9.1f}%")
        _p()

    _p("=" * 62)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, "confidence_stats.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Saved] {txt_path}")

def _parse_lmdb_sample(value) -> np.ndarray:
    """从 pickle bytes 解析出 numpy 数组（复用 run_explainability 的逻辑）"""
    import pickle
    data = pickle.loads(value)
    if isinstance(data, dict):
        data = data.get('sample', data.get('data', data.get('X', data.get('eeg', data))))
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(data, np.ndarray) and data.ndim == 3:
        data = data[np.newaxis, ...]
    return data


def _parse_lmdb_sample_with_label(value):
    """从 pickle bytes 同时解析出 numpy 数组和 label"""
    import pickle
    raw = pickle.loads(value)
    label = None
    if isinstance(raw, dict):
        label = raw.get('label', None)
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, np.ndarray):
            label = int(label.flat[0])
        data = raw.get('sample', raw.get('data', raw.get('X', raw.get('eeg', raw))))
    elif isinstance(raw, (tuple, list)):
        if len(raw) >= 2:
            data, label = raw[0], raw[1]
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = int(label.flat[0])
            elif isinstance(label, (list, tuple)):
                label = int(label[0]) if label else None
        else:
            data = raw[0]
    else:
        data = raw
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    return data, label


def collect_predictions_lmdb_fast(
    adapter,
    dataset_path: str,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    LMDB 专用快速推理：env 只打开/关闭一次，keys 只枚举一次。
    比逐样本 open/close 快数倍。
    """
    import lmdb
    import time

    batch_size = 32
    env = lmdb.open(str(dataset_path), readonly=True, lock=False)
    records = []
    _first_batch_done = False
    _batch_count = 0

    with env.begin() as txn:
        # 一次性枚举所有 key
        keys = [k for k, _ in txn.cursor() if k != b'__keys__']
        total_samples = len(keys)
        print(f"\n[Step 1] Dataset size: {total_samples} samples. Running LMDB fast inference (batch={batch_size})...")
        t_start = time.time()

        batch_tensors = []
        batch_indices = []  # list of (idx, label)

        def _flush_batch(batch_tensors, batch_indices):
            nonlocal _first_batch_done, _batch_count
            _batch_count += 1
            batch = torch.cat(batch_tensors, dim=0)  # (B, ...)
            if not _first_batch_done:
                print(f"  [Inference] batch device: {batch.device}, model device: {next(adapter.model.parameters()).device}")
                _first_batch_done = True
            with torch.no_grad():
                output = adapter.forward(batch)
            for i, (idx, label) in enumerate(batch_indices):
                out = output[i:i+1]
                if out.numel() == 1 or (out.dim() == 2 and out.shape[1] == 1):
                    prob = torch.sigmoid(out).item()
                    pred = int(prob > threshold)
                else:
                    probs = torch.softmax(out, dim=-1)
                    prob = probs.max().item()
                    pred = probs.argmax(dim=-1).item()
                records.append({'index': idx, 'prob': prob, 'pred': pred, 'label': label})

        for idx, key in enumerate(keys):
            try:
                value = txn.get(key)
                if value is None:
                    continue
                data, label = _parse_lmdb_sample_with_label(value)
                data = adapter.prepare_input(data)
                batch_tensors.append(data)
                batch_indices.append((idx, label))

                if len(batch_tensors) >= batch_size:
                    _flush_batch(batch_tensors, batch_indices)
                    batch_tensors = []
                    batch_indices = []

            except Exception as e:
                print(f"  [Warning] Sample {idx} failed: {e}")
                continue

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total_samples}")

        # 处理剩余不足一个 batch 的样本
        if batch_tensors:
            _flush_batch(batch_tensors, batch_indices)

    env.close()
    elapsed = time.time() - t_start
    n_batches = _batch_count
    print(f"  Done. {len(records)} samples in {elapsed:.1f}s, {n_batches} batches (batch_size={batch_size}, {elapsed/max(len(records),1)*1000:.1f}ms/sample).")
    return records


def collect_predictions_mat_fast(
    adapter,
    dataset_path: str,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    MAT 格式快速推理：一次性加载整个 mat 文件到内存，批量推理。
    """
    import scipy.io
    import time

    mat_path = Path(dataset_path)
    if mat_path.is_dir():
        mat_files = list(mat_path.glob('*.mat'))
        if not mat_files:
            raise ValueError(f"No .mat files found in {dataset_path}")
        mat_path = mat_files[0]

    print(f"\n[Step 1] Loading mat file: {mat_path}")
    mat = scipy.io.loadmat(str(mat_path))

    if 'x_data' in mat:
        dataset = mat['x_data']
        labels = mat.get('y_labels', None)
    else:
        keys = [k for k in mat.keys() if not k.startswith('_')]
        dataset = mat[keys[0]]
        labels = None

    total_samples = len(dataset)
    print(f"  Dataset size: {total_samples} samples. Running inference (batch={batch_size})...")

    records = []
    t_start = time.time()

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        try:
            batch_data = []
            for i in range(batch_start, batch_end):
                sample = dataset[i]
                tensor = adapter.prepare_input(sample)
                batch_data.append(tensor)

            batch_tensor = torch.cat(batch_data, dim=0)
            with torch.no_grad():
                output = adapter.model(batch_tensor)

            if output.shape[-1] == 1 or output.numel() == batch_tensor.shape[0]:
                probs = torch.sigmoid(output).squeeze(-1)
                for i, (prob, idx) in enumerate(zip(probs, range(batch_start, batch_end))):
                    p = prob.item()
                    label = int(labels[idx]) if labels is not None else None
                    records.append({'index': idx, 'prob': p, 'pred': int(p > threshold), 'label': label})
            else:
                probs = torch.softmax(output, dim=-1)
                for i, (prob_vec, idx) in enumerate(zip(probs, range(batch_start, batch_end))):
                    p = prob_vec.max().item()
                    pred = prob_vec.argmax().item()
                    label = int(labels[idx]) if labels is not None else None
                    records.append({'index': idx, 'prob': p, 'pred': pred, 'label': label})

            if batch_end % 100 == 0 or batch_end == total_samples:
                print(f"  {batch_end}/{total_samples} done")

        except Exception as e:
            print(f"  [Warning] Batch {batch_start}-{batch_end} failed: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"  Done. {len(records)} samples in {elapsed:.1f}s ({elapsed/max(len(records),1)*1000:.1f}ms/sample).")
    return records


def collect_predictions_pkl_dir_fast(
    adapter,
    dataset_path: str,
    n_samples: int = -1,
    seed: int = 42,
    num_workers: int = 8,
    threshold: float = 0.5,
) -> List[Dict]:
    """pkl 目录快速路径：线程池并行读文件，边读边推理，速度接近 LMDB。"""
    import pickle, random, time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    p = Path(dataset_path)
    all_files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == '.pkl')
    total = len(all_files)

    if n_samples > 0 and n_samples < total:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(total), n_samples))
        files = [all_files[i] for i in chosen]
        indices = chosen
        print(f"\n[Step 1] Dataset size: {total} samples. Randomly sampling {n_samples}. Running inference...")
    else:
        files = all_files
        indices = list(range(total))
        print(f"\n[Step 1] Dataset size: {total} samples. Running inference...")

    def _load_pkl(args):
        idx, f = args
        fname = f.name
        with open(f, 'rb') as fp:
            sample = pickle.load(fp)
        label = None
        if isinstance(sample, dict):
            label = sample.get('label', sample.get('y', None))
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = int(label.flat[0])
            elif isinstance(label, (int, float)):
                label = int(label)
            sample = sample.get('data', sample.get('X', sample.get(
                'eeg', sample.get('signal', list(sample.values())[0]))))
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        return idx, sample, label, fname

    records = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_load_pkl, (idx, f)): idx
                   for idx, f in zip(indices, files)}
        for future in as_completed(futures):
            try:
                idx, sample, label, fname = future.result()
                data = adapter.prepare_input(sample)
                with torch.no_grad():
                    output = adapter.forward(data)

                if output.numel() == 1 or (output.dim() == 2 and output.shape[1] == 1):
                    prob = torch.sigmoid(output).item()
                    pred = int(prob > threshold)
                else:
                    probs = torch.softmax(output, dim=-1)
                    prob = probs.max().item()
                    pred = probs.argmax(dim=-1).item()

                records.append({'index': idx, 'prob': prob, 'pred': pred, 'label': label, 'filename': fname})
            except Exception as e:
                if done < 3:
                    import traceback
                    print(f"  [Error] Sample {futures[future]} failed: {e}")
                    traceback.print_exc()

            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {done}/{len(files)}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done. {len(records)} samples in {elapsed:.1f}s ({elapsed/max(len(records),1)*1000:.1f}ms/sample).")
    return records


def collect_predictions(
    adapter,
    dataset_path: str,
    device: str,
    data_format: Optional[str] = None,
    n_samples: int = -1,
    seed: int = 42,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    遍历数据集，对每个样本做前向推理，收集置信度。
    只做前向传播，不跑 explainability，速度快。
    LMDB 格式自动使用快速路径（env 只开关一次）。
    """
    # 自动检测格式
    if data_format is None:
        p = Path(dataset_path)
        ext = p.suffix.lower()
        if ext == '.npy':
            data_format = 'npy'
        elif ext == '.npz':
            data_format = 'npz'
        elif ext in ('.pt', '.pth'):
            data_format = 'pt'
        elif ext == '.mat':
            data_format = 'mat'
        elif p.is_dir() and (p / 'data.mdb').exists():
            data_format = 'lmdb'
        elif p.is_dir() and list(p.glob('*.mat')):
            data_format = 'mat'
        elif p.is_dir() and list(p.glob('*.pkl')):
            data_format = 'pkl_dir'
        elif p.is_dir():
            data_format = 'pt_dir'
        else:
            data_format = 'lmdb'

    if data_format == 'lmdb':
        return collect_predictions_lmdb_fast(adapter, dataset_path, threshold=threshold)

    if data_format == 'mat':
        return collect_predictions_mat_fast(adapter, dataset_path, threshold=threshold)

    if data_format == 'pkl_dir':
        # n_samples>0 时触发新逻辑：全量扫描，filter_samples 再按置信度取 top-N
        scan_n = -1 if n_samples > 0 else n_samples
        return collect_predictions_pkl_dir_fast(adapter, dataset_path, n_samples=scan_n, seed=seed, threshold=threshold)

    # 非 LMDB/MAT：通用路径
    total_samples = get_dataset_size(dataset_path, data_format)

    # n_samples>0 触发新逻辑：全量扫描（filter_samples 里再按置信度取 top-N）
    # n_samples=-1（默认）：保持原来行为，全量扫描使用全部样本
    indices = list(range(total_samples))
    print(f"\n[Step 1] Dataset size: {total_samples} samples. Running full inference...")

    records = []
    for i, idx in enumerate(indices):
        try:
            data, label = load_from_dataset(dataset_path, idx, data_format)
            data = adapter.prepare_input(data)

            with torch.no_grad():
                output = adapter.model(data)

            if output.numel() == 1 or (output.dim() == 2 and output.shape[1] == 1):
                prob = torch.sigmoid(output).item()
                pred = int(prob > threshold)
            else:
                probs = torch.softmax(output, dim=-1)
                prob = probs.max().item()
                pred = probs.argmax(dim=-1).item()

            records.append({'index': idx, 'prob': prob, 'pred': pred, 'label': label})

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(indices)}")

        except Exception as e:
            if i == 0:
                print(f"  [Error] Sample {idx} failed: {e}")
            continue

    print(f"  Done. Collected {len(records)} valid samples.")
    return records


# ===================== Step 2: 按类别和置信度筛选样本 =====================

def filter_samples(
    records: List[Dict],
    target_class: int,
    conf_threshold: float,
    n_samples: int,
    seed: int = 42,
    is_fp: bool = False,   # True=FP(pred==target_class, label!=target_class), False=TP(pred==label==target_class)
    is_binary: bool = False,
) -> Tuple[List[int], List[float]]:
    """
    从 records 中筛出目标类别的样本。

    is_fp=False (默认): TP，pred==label==target_class
    is_fp=True:         FP，pred==target_class 且 label!=target_class

    Returns:
        selected_indices: 选中的样本索引列表
        selected_confs  : 对应的置信度列表
    """
    sample_type = 'FP' if is_fp else 'TP'
    filtered = []
    for r in records:
        if r['pred'] != target_class:
            continue
        label = r.get('label', None)
        if is_fp:
            # FP: pred==target_class 但 label!=target_class
            if label is None or int(label) == target_class:
                continue
        else:
            # TP: pred==label==target_class
            if label is not None and int(label) != target_class:
                continue

        # 置信度
        if is_binary and target_class == 0:
            conf = 1.0 - r['prob']
        else:
            conf = r['prob']
        if conf >= conf_threshold:
            filtered.append((r['index'], conf))

    print(f"\n[Step 2] target_class={target_class}, type={sample_type}, conf_threshold={conf_threshold}")
    print(f"  Found {len(filtered)} samples")

    if len(filtered) == 0:
        raise ValueError(
            f"No {sample_type} samples found for class={target_class} with conf>={conf_threshold}."
        )

    if n_samples > 0 and len(filtered) > n_samples:
        # 新逻辑：按置信度降序取 top-N
        selected = sorted(filtered, key=lambda x: x[1], reverse=True)[:n_samples]
        print(f"  Selected top-{n_samples} by confidence (got {len(selected)}).")
    else:
        selected = sorted(filtered, key=lambda x: x[1], reverse=True)
        print(f"  Using all {len(selected)} samples (sorted by confidence).")
    selected_indices = [s[0] for s in selected]
    selected_confs   = [s[1] for s in selected]
    return selected_indices, selected_confs


def _extract_subject_id(filename: str) -> str:
    """从 pkl 文件名提取受试者 ID。
    TUAB: aaaaaizd_s001_t001_68.pkl -> aaaaaizd
    TUAB (no _s): aaaaadni_00000001-61.pkl -> aaaaadni
    """
    stem = filename.replace('.pkl', '')
    if '_s' in stem:
        return stem.split('_s')[0]
    parts = stem.split('_')
    if len(parts) >= 2:
        core = parts[1].split('-')[0]
        if core.isdigit():
            if len(parts) >= 3:
                # TUEV: bckg_011_a_-294 -> 011 (parts[0] is class label)
                return core
            else:
                # TUAB variant: aaaaadni_00000001-61 -> aaaaadni (parts[0] is subject)
                return parts[0]
    return stem


def _parse_subject_ids(raw: str):
    """
    解析 --subject-ids 参数，支持两种格式：
      简单: "008,020"  -> ['008', '020']
      按类: "0:008,020;2:015,030" -> {0: ['008','020'], 2: ['015','030']}
    """
    if ':' in raw:
        result = {}
        for part in raw.split(';'):
            cls_str, ids_str = part.split(':', 1)
            result[int(cls_str.strip())] = [s.strip() for s in ids_str.split(',')]
        return result
    else:
        return [s.strip() for s in raw.split(',')]


def _select_fixed_subjects(
    records: List[Dict],
    target_classes: List[int],
    subject_ids: "Dict[int, List[str]] | List[str]",
    conf_threshold: float = 0.0,
    is_binary: bool = False,
) -> Dict[int, List[Dict]]:
    """
    按指定的 subject ID 列表过滤记录，统计 segment 数和平均置信度。

    Args:
        subject_ids: 可以是 List[str]（所有类通用）或 Dict[int, List[str]]（按类指定）
        conf_threshold: 置信度过滤阈值，0 表示不过滤

    Returns:
        {class: [{subject_id, indices, confs, n_segments, label, mean_conf}, ...]}
    """
    from collections import defaultdict

    result = {}
    for cls in target_classes:
        if isinstance(subject_ids, dict):
            cls_subject_ids = subject_ids.get(cls, [])
        else:
            cls_subject_ids = subject_ids

        if not cls_subject_ids:
            print(f"  [Skip] class {cls}: no subject IDs specified")
            continue

        subj_records = defaultdict(list)
        for r in records:
            if r.get('filename') is None:
                continue
            label = r.get('label')
            if label is None or int(label) != cls:
                continue
            if r['pred'] != cls:
                continue
            subj_id = _extract_subject_id(r['filename'])
            if subj_id not in cls_subject_ids:
                continue
            conf = 1.0 - r['prob'] if (is_binary and cls == 0) else r['prob']
            if conf_threshold > 0 and conf < conf_threshold:
                continue
            subj_records[subj_id].append({
                'index': r['index'],
                'conf': conf,
                'filename': r['filename'],
            })

        selected = []
        for subj_id in cls_subject_ids:
            recs = subj_records.get(subj_id, [])
            if not recs:
                print(f"  [Warning] Subject {subj_id} has no TP samples for class {cls}"
                      f"{f' (conf>={conf_threshold})' if conf_threshold > 0 else ''}, skipping")
                continue
            indices = [r['index'] for r in recs]
            confs = [r['conf'] for r in recs]
            selected.append({
                'subject_id': subj_id,
                'indices': indices,
                'confs': confs,
                'n_segments': len(recs),
                'label': cls,
                'mean_conf': np.mean(confs),
            })

        print(f"\n  [Fixed Subject Selection] class {cls}:")
        print(f"  {'Subject ID':<15} {'Segments':>10} {'Mean Conf':>10}")
        print(f"  {'-'*15} {'-'*10} {'-'*10}")
        for s in selected:
            print(f"  {s['subject_id']:<15} {s['n_segments']:>10} {s['mean_conf']:>10.3f}")

        result[cls] = selected

    return result


def select_top_subjects(
    records: List[Dict],
    target_classes: List[int],
    conf_threshold: float,
    n_top: int = 2,
    is_binary: bool = False,
) -> Dict[int, List[Dict]]:
    """
    置信度过滤 + 按受试者分组 + 选取每类 top-N 受试者。

    Returns:
        {class: [{subject_id, indices, confs, n_segments, label}, ...]}
    """
    from collections import defaultdict

    result = {}
    for cls in target_classes:
        subj_records = defaultdict(list)
        for r in records:
            if r.get('filename') is None:
                continue
            label = r.get('label')
            if label is None or int(label) != cls:
                continue
            if r['pred'] != cls:
                continue
            if is_binary and cls == 0:
                conf = 1.0 - r['prob']
            else:
                conf = r['prob']
            if conf >= conf_threshold:
                subj_id = _extract_subject_id(r['filename'])
                subj_records[subj_id].append({
                    'index': r['index'],
                    'conf': conf,
                    'filename': r['filename'],
                })

        ranked = sorted(subj_records.items(), key=lambda x: len(x[1]), reverse=True)

        print(f"\n  [Subject Selection] class {cls}: "
              f"{len(subj_records)} subjects passed filter (conf>={conf_threshold})")
        print(f"  {'Subject ID':<15} {'Segments':>10} {'Mean Conf':>10}")
        print(f"  {'-'*15} {'-'*10} {'-'*10}")
        for subj_id, recs in ranked[:min(10, len(ranked))]:
            mean_conf = np.mean([r['conf'] for r in recs])
            marker = " <--" if ranked.index((subj_id, recs)) < n_top else ""
            print(f"  {subj_id:<15} {len(recs):>10} {mean_conf:>10.3f}{marker}")

        selected = []
        for subj_id, recs in ranked[:n_top]:
            indices = [r['index'] for r in recs]
            confs = [r['conf'] for r in recs]
            selected.append({
                'subject_id': subj_id,
                'indices': indices,
                'confs': confs,
                'n_segments': len(recs),
                'label': cls,
                'mean_conf': np.mean(confs),
            })

        result[cls] = selected

    return result

def _plot_grand_avg_waveform_with_attribution(
    raw_signals: List[np.ndarray],
    grand_temporal_importance: np.ndarray,
    channel_names: List[str],
    n_patches: int,
    patch_size: int,
    title: str = "Grand Avg Waveform + Temporal Attribution",
    output_dir: str = ".",
    save: bool = True,
    show: bool = False,
):
    """群体平均波形 + 时间归因叠加图"""
    import matplotlib.pyplot as plt

    signals = np.stack(raw_signals, axis=0)
    if signals.ndim == 4:
        # (N, channels, patches, features) -> (N, channels, total_time)
        N, C, P, F = signals.shape
        signals = signals.reshape(N, C, P * F)
    else:
        N, C, T = signals.shape

    grand_avg = signals.mean(axis=0)  # (C, total_time)
    total_time = grand_avg.shape[1]

    # 将 patch-level 归因展开到 sample level
    temporal_expanded = np.repeat(grand_temporal_importance, patch_size)[:total_time]
    temporal_norm = (temporal_expanded - temporal_expanded.min()) / (temporal_expanded.max() - temporal_expanded.min() + 1e-8)

    n_ch = min(len(channel_names), C)
    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(14, n_ch * 0.8 + 2),
                             gridspec_kw={'height_ratios': [1] * n_ch + [0.6]}, sharex=True)
    if n_ch + 1 == 1:
        axes = [axes]

    time_axis = np.arange(total_time)

    for ch in range(n_ch):
        ax = axes[ch]
        ax.plot(time_axis, grand_avg[ch], color='k', linewidth=0.7)
        ax.fill_between(time_axis, grand_avg[ch].min(), grand_avg[ch].max(),
                        where=temporal_norm > 0.7, alpha=0.3, color='red')
        ax.set_ylabel(channel_names[ch], fontsize=7, rotation=0, ha='right', va='center')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for p in range(1, n_patches):
            ax.axvline(x=p * patch_size, color='#dddddd', linewidth=0.3)

    # 底部：时间归因曲线
    ax_bottom = axes[-1]
    ax_bottom.bar(range(n_patches), grand_temporal_importance,
                  color=['#d32f2f' if v > np.percentile(grand_temporal_importance, 75) else '#90caf9'
                         for v in grand_temporal_importance], edgecolor='none')
    ax_bottom.set_xlabel('Patch Index')
    ax_bottom.set_ylabel('Attribution', fontsize=8)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=10, y=0.995)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'grand_avg_waveform_attribution.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"    [Saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_top_sample_waveforms(
    raw_signals: List[np.ndarray],
    combineds: List[np.ndarray],
    valid_confs: List[float],
    channel_names: List[str],
    n_top_samples: int = 3,
    title_prefix: str = "",
    output_dir: str = ".",
    show: bool = False,
):
    """挑选 top-N 高置信样本，画 waveform + attribution 着色图"""
    from explainability.visualizer import EEGExplainabilityVisualizer

    os.makedirs(output_dir, exist_ok=True)
    viz = EEGExplainabilityVisualizer(dpi=150, show=False)

    ranked_idx = np.argsort(valid_confs)[::-1][:n_top_samples]

    for rank, idx in enumerate(ranked_idx):
        waveform = raw_signals[idx]
        attribution = combineds[idx]
        conf = valid_confs[idx]

        sample_title = f"{title_prefix}\nSample rank={rank+1} | conf={conf:.3f}"
        save_path = os.path.join(output_dir, f'sample_rank{rank+1}_conf{conf:.3f}.png')

        viz.plot_waveform_with_heatmap(
            waveform=waveform,
            attribution=attribution,
            channel_names=channel_names,
            title=sample_title,
            save_path=save_path,
            show=show,
        )
        print(f"    [Saved] {save_path}")


def run_explainability_batch(
    adapter,
    explainer,
    dataset_path: str,
    selected_indices: List[int],
    selected_confs: List[float],
    device: str,
    target_class: int = 0,
    data_format: Optional[str] = None,
    index_to_source: Optional[Dict] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    对选中的样本逐个跑 explainability，收集 raw_signal 和 combined。

    Returns:
        raw_signals : list of (n_channels, signal_length)
        combineds   : list of (n_channels, n_patches)
    """
    raw_signals = []
    combineds = []
    valid_confs = []
    n = len(selected_indices)
    print(f"\n[Step 3] Running explainability on {n} selected samples...")
    t_start = time.time()

    # mat 格式预加载到内存
    _mat_dataset = None
    _pkl_files = None
    p = Path(dataset_path)
    if p.is_dir() and list(p.glob('*.mat')):
        import scipy.io
        mat_file = list(p.glob('*.mat'))[0]
        mat = scipy.io.loadmat(str(mat_file))
        if 'x_data' in mat:
            _mat_dataset = mat['x_data']
        else:
            keys = [k for k in mat.keys() if not k.startswith('_')]
            _mat_dataset = mat[keys[0]] if keys else None
    elif p.is_dir() and data_format == 'pkl_dir':
        _pkl_files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == '.pkl')
    elif p.is_dir():
        _dir_files = sorted(f for f in p.iterdir() if f.is_file())

    for i, (idx, conf) in enumerate(zip(selected_indices, selected_confs)):
        try:
            # 多数据集时，还原为原始路径和索引
            if index_to_source and idx in index_to_source:
                _src_path, _src_idx = index_to_source[idx]
                data, _ = load_from_dataset(_src_path, _src_idx, data_format)
            elif _mat_dataset is not None:
                sample = _mat_dataset[idx]
                if sample.ndim == 2:
                    data = sample[np.newaxis, ...]
                elif sample.ndim == 3:
                    data = sample[np.newaxis, ...]
                else:
                    data = sample
            elif _pkl_files is not None and idx < len(_pkl_files):
                import pickle
                with open(_pkl_files[idx], 'rb') as fp:
                    sample = pickle.load(fp)
                if isinstance(sample, dict):
                    sample = sample.get('data', sample.get('X', sample.get(
                        'eeg', sample.get('signal', list(sample.values())[0]))))
                if isinstance(sample, torch.Tensor):
                    data = sample.numpy()
                elif isinstance(sample, np.ndarray):
                    data = sample
                else:
                    data = np.array(sample)
                if data.ndim == 2:
                    data = data[np.newaxis, ...]  # (C, T) -> (1, C, T)
                elif data.ndim == 3:
                    data = data[np.newaxis, ...]  # (C, P, ps) -> (1, C, P, ps)
            else:
                data, _ = load_from_dataset(dataset_path, idx, data_format)

            # 保存原始信号（preprocess 之前）
            raw = data.squeeze(0) if data.ndim == 4 else data
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = raw.squeeze(0)          # (1, C, T) -> (C, T)
            elif raw.ndim == 3:
                raw = raw.reshape(raw.shape[0], -1)  # (C, P, F) -> (C, P*F)
            raw_signal = raw.copy() if isinstance(raw, np.ndarray) else raw

            # preprocess → tensor (统一走 prepare_input 保证 4D)
            data_tensor = adapter.prepare_input(data)
            data_clean = data_tensor.clone().detach()

            # 跑 explainability，传入 target_class 让归因方向正确
            result = explainer.explain(data_clean, target=target_class)

            raw_signals.append(raw_signal)
            combineds.append(result['combined'])
            valid_confs.append(conf)

            elapsed = time.time() - t_start
            print(f"  {i + 1}/{n} done  ({elapsed:.1f}s, ~{elapsed/(i+1)*1000:.0f}ms/sample)")

        except Exception as e:
            print(f"  [Warning] Sample {idx} failed: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"  Collected {len(raw_signals)} valid explainability results in {elapsed:.1f}s.")
    return raw_signals, combineds, valid_confs


# ===================== 通道重要度（均值） =====================

def compute_grand_channel_importance(
    combineds: List[np.ndarray],
    top_k: int,
    confs: Optional[List[float]] = None,
) -> np.ndarray:
    """
    跨样本计算群体通道重要度。

    流程：
      1. 每个样本：signed_mean 聚合 patch → 通道（不做逐样本归一化）
      2. 跨样本：取均值（保留完整排序梯度，N 大时对离群值不敏感）
      3. 最终归一化到 [-1, 1]

    返回值 grand_channel_importance (n_channels,)：
      正值 = 支持分类，负值 = 反对分类。
    """
    n_samples = len(combineds)
    n_channels = combineds[0].shape[0]

    channel_importance_all = np.stack(
        [np.mean(c, axis=-1) for c in combineds], axis=0
    )  # (n_samples, n_channels)

    grand_mean = np.mean(channel_importance_all, axis=0)  # (n_channels,)

    max_abs = np.max(np.abs(grand_mean))
    if max_abs > 1e-8:
        grand_channel_importance = grand_mean / max_abs
    else:
        grand_channel_importance = grand_mean

    # top-K：正贡献最强的通道
    pos_indices = np.where(grand_channel_importance > 0)[0]
    if len(pos_indices) > 0:
        top_pos = pos_indices[np.argsort(grand_channel_importance[pos_indices])[::-1]][:top_k]
    else:
        top_pos = np.argsort(grand_channel_importance)[::-1][:top_k]

    # top-K 负贡献
    neg_indices = np.where(grand_channel_importance < 0)[0]
    if len(neg_indices) > 0:
        top_neg = neg_indices[np.argsort(grand_channel_importance[neg_indices])][:top_k]
    else:
        top_neg = np.array([], dtype=int)

    n_pos = np.sum(grand_channel_importance > 0)
    n_neg = np.sum(grand_channel_importance < 0)
    print(f"  [Channel Importance] mean-based, "
          f"positive: {n_pos}/{n_channels}, negative: {n_neg}/{n_channels}")

    return grand_channel_importance, top_pos, top_neg


def compute_grand_temporal_importance_voting(
    combineds: List[np.ndarray],
    top_k: int,
    confs: Optional[List[float]] = None,
    temporal_aggregation: str = 'signed_mean',
) -> np.ndarray:
    """
    跨样本 voting 机制计算群体时间（patch）重要度。

    每个样本基于自身 combined.mean(axis=0) 选出正贡献 top_k patches 投票，
    最终返回 grand_temporal_importance (n_patches,)。
    """
    n_samples = len(combineds)
    n_patches = combineds[0].shape[1]

    def _agg(c):
        if temporal_aggregation == 'relu_mean':
            return np.mean(np.maximum(c, 0), axis=0)
        elif temporal_aggregation == 'abs_mean':
            return np.mean(np.abs(c), axis=0)
        else:  # signed_mean
            return np.mean(c, axis=0)

    patch_importance_all = np.stack(
        [_agg(c) for c in combineds], axis=0
    )  # (n_samples, n_patches)

    # 归一化每个样本
    for i in range(n_samples):
        max_val = np.abs(patch_importance_all[i]).max()
        if max_val > 0:
            patch_importance_all[i] /= max_val

    # 加权均值
    weights = np.array(confs, dtype=float) if confs is not None else np.ones(n_samples)
    grand_mean = np.average(patch_importance_all, axis=0, weights=weights)

    # Voting
    vote_counts = np.zeros(n_patches, dtype=int)
    for i in range(n_samples):
        pos_mask = patch_importance_all[i] > 0
        pos_indices = np.where(pos_mask)[0]
        if len(pos_indices) == 0:
            continue
        sample_top = pos_indices[np.argsort(patch_importance_all[i][pos_indices])[::-1]][:top_k]
        vote_counts[sample_top] += 1

    print(f"  [Temporal Voting] vote_counts: {list(vote_counts)}, "
          f"positive mean patches: {np.sum(grand_mean > 0)}/{n_patches}")

    grand_temporal_importance = grand_mean.copy()
    pos_mask = grand_mean > 0
    if pos_mask.any():
        vote_norm = vote_counts[pos_mask] / max(vote_counts.max(), 1)
        mean_norm = grand_mean[pos_mask] / (grand_mean[pos_mask].max() + 1e-8)
        grand_temporal_importance[pos_mask] = 0.6 * vote_norm + 0.4 * mean_norm

    return grand_temporal_importance


# ===================== 主函数 =====================

def run_population_analysis(
    model_type: str,
    config: dict,
    methods: List[str],           # 支持多个方法
    dataset_path,                 # str 或 List[str]
    checkpoint_path: Optional[str] = None,
    task: Optional[str] = None,
    device: str = 'cuda',
    target_classes: Optional[List[int]] = None,  # None = 所有类别
    tp_conf_threshold: float = 0.7,   # TP 置信度门槛
    fp_conf_threshold: float = 0.8,   # FP 置信度门槛（高置信误判更有诊断价值）
    n_samples: int = -1,
    top_k: int = 5,
    output_dir: str = './population_results',
    data_format: Optional[str] = None,
    seed: int = 42,
    run_patch_band_corr: bool = False,  # 默认关闭，分析意义与 wavelet heatmap 重叠
    band_methods: Optional[List[str]] = None,
    band_baseline: str = 'auto',
    skip_band_attribution: bool = False,
    n_workers: int = 1,
    per_subject: bool = False,
    n_top_subjects: int = 2,
    subject_ids=None,
    **model_kwargs,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建模型和适配器（只创建一次）
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
    print(f"  Model moved to: {next(model.parameters()).device}")

    adapter = ModelAdapterRegistry.create(
        name=model_type,
        model=model,
        config=config,
        task=task,
        device=device,
    )
    channel_names = adapter.get_channel_names()
    print(f"  Channels: {channel_names[:5]}... ({len(channel_names)} total)")

    # 2. 全量推理（支持多数据路径）
    _cls_threshold = 0.6394 if model_type == 'biot' else 0.5
    if isinstance(dataset_path, list):
        all_records = []
        path_offsets = {}  # path -> index offset，用于后续按 index 加载数据
        offset = 0
        for dp in dataset_path:
            print(f"\n  [Dataset] {dp}")
            recs = collect_predictions(
                adapter=adapter, dataset_path=dp, device=device,
                data_format=data_format, n_samples=-1,
                threshold=_cls_threshold,
            )
            for r in recs:
                r['_source_path'] = dp
                r['_source_index'] = r['index']
                r['index'] = r['index'] + offset
            path_offsets[dp] = offset
            offset += len(recs)
            all_records.extend(recs)
        records = all_records
        _multi_dataset = True
        print(f"\n  Total: {len(records)} samples from {len(dataset_path)} datasets")
    else:
        _multi_dataset = False
        records = collect_predictions(
            adapter=adapter,
            dataset_path=dataset_path,
            device=device,
            data_format=data_format,
            n_samples=n_samples,
            threshold=_cls_threshold,
        )

    # 多数据集时：建全局 index → (source_path, source_index) 映射
    # 单数据集时：dataset_path 不变，index 就是原始 index
    _index_to_source = {}
    if _multi_dataset:
        for r in records:
            _index_to_source[r['index']] = (r['_source_path'], r['_source_index'])
        # dataset_path 设为第一个，供不支持多路径的旧函数降级用
        dataset_path = dataset_path[0]

    # 确定要分析的类别
    if target_classes is None:
        target_classes = sorted(set(r['pred'] for r in records))
        print(f"\n  Auto-detected classes: {target_classes}")

    # 自动判断二分类还是多分类
    all_labels = [r.get('label') for r in records if r.get('label') is not None]
    n_classes = len(set(int(l) for l in all_labels)) if all_labels else len(target_classes)
    is_binary = (n_classes == 2)
    print(f"  Task type: {'binary' if is_binary else 'multiclass'} ({n_classes} classes)")

    # ===================== 自动检测 label/pred 偏移 =====================
    if all_labels:
        pred_set = set(r['pred'] for r in records)
        label_set = set(int(l) for l in all_labels)

        # 模型有效输出范围: 多分类 {0..num_classes-1}, 二分类(sigmoid) {0,1}
        num_classes = config.get('num_classes', None)
        if num_classes is not None:
            max_valid_pred = max(1, num_classes - 1)
        else:
            max_valid_pred = max(pred_set) if pred_set else 0

        need_shift = False

        # 核心判断: label 中存在超出模型输出范围的值，且不含 0 → 1-indexed
        if 0 not in label_set and max(label_set) > max_valid_pred:
            need_shift = True

        # 回退: 无交集检测（兜底，适用于 num_classes 未知的情况）
        if not need_shift and len(pred_set & label_set) == 0 and len(pred_set) > 0:
            shifted_label_set = set(l - 1 for l in label_set)
            if pred_set == shifted_label_set or pred_set.issubset(shifted_label_set):
                need_shift = True

        if need_shift:
            print(f"\n  [Auto-fix] Detected label offset: labels are 1-indexed ({sorted(label_set)}), "
                  f"preds are 0-indexed ({sorted(pred_set)}). Shifting labels by -1.")
            for r in records:
                if r.get('label') is not None:
                    r['label'] = int(r['label']) - 1
            all_labels = [r.get('label') for r in records if r.get('label') is not None]
            if target_classes is not None:
                target_classes = sorted(set(r['pred'] for r in records))
    # =====================================================================

    # ===================== 置信度分布统计（保存到 txt）=====================
    _debug_confidence_stats(records, target_classes, is_binary,
                            output_dir=output_dir,
                            cls_threshold=_cls_threshold)

    # ===================== PSD 预缓存：确保每类有 PSD_N_SAMPLES 个信号 =====================
    # 同时为 class_permute 基线预备对立类信号（复用同一份缓存，避免重复加载）
    skip_band = skip_band_attribution or band_methods is None
    BAND_DONOR_N_SAMPLES = 50
    _need_donors = (not skip_band and band_baseline in ('auto', 'class_permute')
                    and len(target_classes) >= 2)
    _psd_supplement = {}  # {class: list of raw signals} 补充样本
    for cls in target_classes:
        # 统计该类正确预测且满足置信度的样本数
        cls_conf_thr = tp_conf_threshold
        n_correct = 0
        for r in records:
            if r['pred'] != cls:
                continue
            label = r.get('label', None)
            if label is not None and int(label) != cls:
                continue
            conf = 1.0 - r['prob'] if (is_binary and cls == 0) else r['prob']
            if conf >= cls_conf_thr:
                n_correct += 1

        # 归因阶段实际缓存的数量受 n_samples 限制
        n_will_cache = n_correct if n_samples <= 0 else min(n_correct, n_samples)
        # PSD 补充量：凑满 PSD_N_SAMPLES
        # class_permute 补充量：始终至少 BAND_DONOR_N_SAMPLES（因为处理第一个类时 _psd_cache 还没有其他类的数据）
        psd_deficit = max(0, PSD_N_SAMPLES - n_will_cache)
        donor_minimum = BAND_DONOR_N_SAMPLES if _need_donors else 0
        deficit = max(psd_deficit, donor_minimum)
        if deficit > 0:
            # 补充样本优先级：剩余高置信度TP > 低置信度TP > 随机原始样本
            # 收集所有正确预测（不限阈值），按置信度排序
            all_correct_with_conf = []
            for r in records:
                if r['pred'] != cls:
                    continue
                label = r.get('label', None)
                if label is not None and int(label) != cls:
                    continue
                conf = 1.0 - r['prob'] if (is_binary and cls == 0) else r['prob']
                all_correct_with_conf.append((r['index'], conf))
            all_correct_with_conf.sort(key=lambda x: x[1], reverse=True)

            # 归因阶段取 top n_will_cache 个（高置信度），补充从第 n_will_cache+1 开始
            remaining_tp = [idx for idx, _ in all_correct_with_conf[n_will_cache:]]

            # 如果所有TP（含低置信度）不够deficit，再从该类随机样本补
            if len(remaining_tp) >= deficit:
                supplement_indices = remaining_tp[:deficit]
                n_from_tp = deficit
                n_from_random = 0
            else:
                supplement_indices = list(remaining_tp)
                n_from_tp = len(remaining_tp)
                # 随机补充：该类所有样本中排除已用的正确预测
                used_indices = set(idx for idx, _ in all_correct_with_conf)
                cls_other_indices = [r['index'] for r in records
                                     if r.get('label') is not None and int(r['label']) == cls
                                     and r['index'] not in used_indices]
                rng_sup = random.Random(seed)
                rng_sup.shuffle(cls_other_indices)
                still_need = deficit - len(remaining_tp)
                supplement_indices.extend(cls_other_indices[:still_need])
                n_from_random = min(still_need, len(cls_other_indices))

            print(f"\n  [PSD Pre-cache] class {cls}: TP(all)={len(all_correct_with_conf)}, "
                  f"TP(conf>={cls_conf_thr})={n_correct}, will_cache={n_will_cache} (n_samples={n_samples}), "
                  f"deficit={deficit}, supplement: {n_from_tp} remaining TP + {n_from_random} random")
            sigs = []
            for idx in supplement_indices:
                try:
                    if _multi_dataset and idx in _index_to_source:
                        _src_path, _src_idx = _index_to_source[idx]
                        data, _ = load_from_dataset(_src_path, _src_idx, data_format)
                    else:
                        data, _ = load_from_dataset(dataset_path, idx, data_format)
                    if isinstance(data, torch.Tensor):
                        raw = data.cpu().numpy()
                    elif isinstance(data, np.ndarray):
                        raw = data
                    else:
                        continue
                    if raw.ndim == 4:
                        raw = raw.squeeze(0)
                    if raw.ndim == 3:
                        if raw.shape[0] == 1:
                            raw = raw.squeeze(0)
                        else:
                            raw = raw.reshape(raw.shape[0], -1)
                    sigs.append(raw)
                except Exception:
                    continue
            _psd_supplement[cls] = sigs
            print(f"    Loaded {len(sigs)} supplement signals for PSD")
        else:
            print(f"\n  [PSD Pre-cache] class {cls}: TP={n_correct}, will_cache={n_will_cache} >= {PSD_N_SAMPLES}, no supplement needed")
            _psd_supplement[cls] = []
    # =====================================================================

    # ===================== 状态任务：计算公共频谱基线（暂时跳过，时频分析已注释） =====================
    _spectral_baseline = None
    _task_type = infer_task_type(task) if task else 'event'
  
 
    def _run_one_group(target_class, is_fp, method, explainer):
        sample_type = 'FP' if is_fp else 'TP'
        conf_thr = fp_conf_threshold if is_fp else tp_conf_threshold
        try:
            selected_indices, selected_confs = filter_samples(
                records=records,
                target_class=target_class,
                conf_threshold=conf_thr,
                n_samples=n_samples,
                seed=seed,
                is_fp=is_fp,
                is_binary=is_binary,
            )
        except ValueError as e:
            print(f"  [Skip] {e}")
            return None, None

        raw_signals, combineds, valid_confs = run_explainability_batch(
            adapter=adapter,
            explainer=explainer,
            dataset_path=dataset_path,
            selected_indices=selected_indices,
            selected_confs=selected_confs,
            device=device,
            target_class=target_class,
            data_format=data_format,
            index_to_source=_index_to_source if _multi_dataset else None,
        )

        # IG 会把模型保持在 float64，faithfulness 需要 float32，这里统一恢复
        adapter.model.float()

        if len(raw_signals) == 0:
            print(f"  [Skip] No valid samples for {sample_type}, method={method}, class={target_class}")
            return None, None

        spectral_output_dir = _build_population_output_dir(
            base_dir=output_dir,
            task=task,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            method=method,
            target_class=target_class,
            sample_type=sample_type,
        )

        # 计算群体通道重要度（中位数）
        grand_channel_importance, top_pos, top_neg = compute_grand_channel_importance(
            combineds=combineds,
            top_k=top_k,
        )

        n_valid = len(raw_signals)
        conf_thr = fp_conf_threshold if is_fp else tp_conf_threshold
        low_sample_warning = n_valid < 5
        # 统一标题前缀：模型 + 任务 + 类别 + 归因方法 + 样本数 + 置信度
        title_prefix = (
            f"{model_type.upper()} | {task} | class {target_class} ({sample_type}) | "
            f"{method.upper()} | n={n_valid} samples  conf≥{conf_thr}"
            + ("  ⚠ LOW SAMPLE" if low_sample_warning else "")
        )
        # subtitle 供 topomap JSON 元数据解析用（保持原格式）
        subtitle = (
            f"{sample_type}  |  class {target_class}  |  method: {method.upper()}  |  "
            f"n={n_valid}  conf≥{conf_thr}"
            + ("  ⚠ LOW SAMPLE COUNT" if low_sample_warning else "")
        )
        if low_sample_warning:
            print(f"  [Warning] Only {n_valid} samples for {sample_type} class={target_class} "
                  f"(conf>={conf_thr}). Results have low statistical reliability.")

        # 根据 task_type 路由分析
        task_type = config.get('task_type', 'state')
        os.makedirs(spectral_output_dir, exist_ok=True)

        faithfulness_result = None
        temporal_faithfulness_result = None
        grand_temporal_importance = None

        if task_type == 'event':
            # 事件检测任务：spatial faithfulness (通道级，用于主表) + temporal (补充)
            faithfulness_result = spatial_faithfulness(
                raw_signals=raw_signals,
                adapter=adapter,
                grand_channel_importance=grand_channel_importance,
                channel_names=channel_names,
                top_k=top_k,
                output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                save=True,
                show=False,
                title_prefix=title_prefix,
            )

            # Single-deletion + Spearman ρ (spatial / channel-level)
            single_deletion_result = spatial_single_deletion(
                raw_signals=raw_signals,
                adapter=adapter,
                grand_channel_importance=grand_channel_importance,
                channel_names=channel_names,
                output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                save=True,
            )
            if faithfulness_result:
                faithfulness_result['spearman_rho'] = single_deletion_result['spearman_rho']
                faithfulness_result['spearman_p'] = single_deletion_result['spearman_p']
                faithfulness_result['single_deletion_deltas'] = single_deletion_result['deltas']

            # Temporal importance + temporal faithfulness (AOPC + Spearman)
            grand_temporal_importance = compute_grand_temporal_importance_voting(
                combineds=combineds,
                top_k=top_k,
                confs=valid_confs,
            )

            _model_info = adapter.get_model_info()
            _patch_size = _model_info.get('patch_size', config.get('patch_size', 200))
            _patch_stride = _model_info.get('patch_stride', _patch_size)
            _n_patches = combineds[0].shape[1]

            temporal_faithfulness_result = temporal_faithfulness(
                raw_signals=raw_signals,
                adapter=adapter,
                grand_temporal_importance=grand_temporal_importance,
                n_patches=_n_patches,
                patch_size=_patch_size,
                patch_stride=_patch_stride,
                top_k=top_k,
                output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                save=True,
                show=False,
                title_prefix=title_prefix,
            )

            temporal_single_del_result = temporal_single_deletion(
                raw_signals=raw_signals,
                adapter=adapter,
                grand_temporal_importance=grand_temporal_importance,
                n_patches=_n_patches,
                patch_size=_patch_size,
                patch_stride=_patch_stride,
                output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                save=True,
            )
            if temporal_faithfulness_result:
                temporal_faithfulness_result['spearman_rho'] = temporal_single_del_result['spearman_rho']
                temporal_faithfulness_result['spearman_p'] = temporal_single_del_result['spearman_p']
                temporal_faithfulness_result['single_deletion_deltas'] = temporal_single_del_result['deltas']

        else:
            # 状态识别任务：spatial importance + topomap + spatial faithfulness
            from explainability.spectral_attribution import _plot_grand_avg_topomap
            _plot_grand_avg_topomap(
                grand_channel_importance=grand_channel_importance,
                channel_names=channel_names,
                top_pos=top_pos,
                top_neg=top_neg,
                n_samples=n_valid,
                output_dir=spectral_output_dir,
                subtitle=subtitle,
                title_prefix=title_prefix,
                save=True,
                show=False,
            )

            faithfulness_result = spatial_faithfulness(
                raw_signals=raw_signals,
                adapter=adapter,
                grand_channel_importance=grand_channel_importance,
                channel_names=channel_names,
                top_k=top_k,
                output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                save=True,
                show=False,
                title_prefix=title_prefix,
            )

            # Single-deletion + Spearman ρ (spatial)
            single_deletion_result = spatial_single_deletion(
                raw_signals=raw_signals,
                adapter=adapter,
                grand_channel_importance=grand_channel_importance,
                channel_names=channel_names,
                output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                save=True,
            )
            if faithfulness_result:
                faithfulness_result['spearman_rho'] = single_deletion_result['spearman_rho']
                faithfulness_result['spearman_p'] = single_deletion_result['spearman_p']
                faithfulness_result['single_deletion_deltas'] = single_deletion_result['deltas']

        # 频段归因（仅 TP，保存在同一目录下）
        if not is_fp and not skip_band:
            try:
                band_out = os.path.join(spectral_output_dir, 'band_attribution')
                os.makedirs(band_out, exist_ok=True)
                # class_permute: 收集所有其他类的信号作为置换源
                # 优先从归因阶段缓存(_psd_cache)取，不足则用预加载补充(_psd_supplement)
                other_class_signals = []
                if _need_donors:
                    for cls in target_classes:
                        if cls == target_class:
                            continue
                        if cls in _psd_cache:
                            other_class_signals.extend(_psd_cache[cls])
                        if cls in _psd_supplement:
                            other_class_signals.extend(_psd_supplement[cls])
                band_result = grand_band_attribution(
                    raw_signals=raw_signals,
                    adapter=adapter,
                    target_class=target_class,
                    fs=float(config.get('fs', 200)),
                    methods=band_methods,
                    baseline_mode=band_baseline,
                    model_type=model_type,
                    channel_names=channel_names,
                    grand_channel_importance=None,
                    top_k=top_k,
                    output_dir=band_out,
                    save=True,
                    show=False,
                    seed=seed,
                    n_workers=n_workers,
                    other_class_signals=other_class_signals if other_class_signals else None,
                )
                if band_result and 'results' in band_result:
                    first_method_key = list(band_result['results'].keys())[0]
                    band_top = band_result['results'][first_method_key].get('top_k_channels', [])
                    for ch_name, _ in band_top:
                        all_top_channels.add(ch_name)
            except Exception as e:
                import traceback
                print(f"    [Warning] Band attribution failed: {e}")
                traceback.print_exc()

        _write_run_info(
            spectral_output_dir,
            Model=model_type,
            Checkpoint=checkpoint_path or 'N/A',
            Task=task,
            Method=method,
            Dataset=dataset_path,
            TargetClass=target_class,
            SampleType=sample_type,
            NSamples=n_valid,
            ConfThreshold=conf_thr,
            _topk_channels=[(channel_names[i], grand_channel_importance[i]) for i in top_pos],
        )

        # 保存结构化结果 JSON（供 compare_models.py 读取）
        import json
        summary_json = {
            'model': model_type,
            'task': task,
            'method': method,
            'target_class': target_class,
            'sample_type': sample_type,
            'n_samples': n_valid,
            'task_type': task_type,
            'channel_names': channel_names,
            'grand_channel_importance': grand_channel_importance.tolist(),
        }
        if grand_temporal_importance is not None:
            summary_json['grand_temporal_importance'] = grand_temporal_importance.tolist()
        if faithfulness_result:
            summary_json['faithfulness'] = {
                'aopc_gain': faithfulness_result.get('aopc_gain'),
                'aopc_attribution': faithfulness_result.get('aopc_attribution'),
                'aopc_random': faithfulness_result.get('aopc_random'),
                'spearman_rho': faithfulness_result.get('spearman_rho'),
                'spearman_p': faithfulness_result.get('spearman_p'),
                'single_deletion_deltas': faithfulness_result.get('single_deletion_deltas'),
            }
        if temporal_faithfulness_result:
            summary_json['temporal_faithfulness'] = {
                'aopc_gain': temporal_faithfulness_result.get('aopc_gain'),
                'aopc_attribution': temporal_faithfulness_result.get('aopc_attribution'),
                'aopc_random': temporal_faithfulness_result.get('aopc_random'),
                'spearman_rho': temporal_faithfulness_result.get('spearman_rho'),
                'spearman_p': temporal_faithfulness_result.get('spearman_p'),
                'single_deletion_deltas': temporal_faithfulness_result.get('single_deletion_deltas'),
            }
        summary_path = os.path.join(spectral_output_dir, 'population_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [Saved] {summary_path}")

        print(f"  Saved to: {spectral_output_dir}")

        top_channel_names = [channel_names[i] for i in top_pos] if channel_names else None
        return raw_signals, top_channel_names

    # 3. 对每个类别、每个方法跑分析
    raw_signals_by_class = {}
    _psd_cache = {}  # 缓存归因阶段的全部信号供 PSD 使用
    all_top_channels = set()
    skip_time = methods is None

    if skip_time and skip_band:
        print("\n[Info] No --method or --band-methods specified, only PSD will be generated.")

    # ===================== 受试者级分析模式 =====================
    if per_subject:
        has_filenames = any(r.get('filename') for r in records)
        if not has_filenames:
            print("\n  [Warning] --per-subject requires pkl_dir format with filenames. Falling back to normal mode.")
            per_subject = False

    if per_subject:
        if subject_ids:
            top_subjects = _select_fixed_subjects(
                records=records,
                target_classes=target_classes,
                subject_ids=subject_ids,
                conf_threshold=tp_conf_threshold,
                is_binary=is_binary,
            )
            print(f"\n{'='*60}")
            print(f"  Per-Subject Analysis Mode (fixed subject IDs)")
            if isinstance(subject_ids, dict):
                for cls, ids in subject_ids.items():
                    print(f"  class {cls}: {', '.join(ids)}")
            else:
                print(f"  Subjects: {', '.join(subject_ids)}")
            print(f"{'='*60}")
        else:
            top_subjects = select_top_subjects(
                records=records,
                target_classes=target_classes,
                conf_threshold=tp_conf_threshold,
                n_top=n_top_subjects,
                is_binary=is_binary,
            )
            print(f"\n{'='*60}")
            print(f"  Per-Subject Analysis Mode")
            print(f"  Top {n_top_subjects} subjects per class selected")
            print(f"{'='*60}")

        for target_class in target_classes:
            subjects = top_subjects.get(target_class, [])
            if not subjects:
                print(f"\n  [Skip] class {target_class}: no subjects passed confidence filter")
                continue

            for subj_info in subjects:
                subj_id = subj_info['subject_id']
                subj_indices = subj_info['indices']
                subj_confs = subj_info['confs']
                n_seg = subj_info['n_segments']
                mean_conf = subj_info['mean_conf']

                print(f"\n{'='*50}")
                print(f"  Subject: {subj_id} | class {target_class} | "
                      f"{n_seg} segments | mean_conf={mean_conf:.3f}")
                print(f"{'='*50}")

                if not skip_time:
                    for method in methods:
                        print(f"\n  Method: {method.upper()}")
                        try:
                            model.eval()
                            extra_kwargs = {}
                            if method in ('ig', 'integrated_gradients'):
                                extra_kwargs['n_steps'] = 20
                            explainer = ExplainabilityRegistry.create(
                                method, adapter, device=device, **extra_kwargs
                            )

                            # 构建受试者级输出目录
                            ckpt_stem = Path(checkpoint_path).stem if checkpoint_path else 'no_ckpt'
                            spectral_output_dir = os.path.join(
                                output_dir, 'population', task,
                                f'{model_type}_{ckpt_stem}', method,
                                f'class_{target_class}_TP',
                                f'subject_{subj_id}',
                            )

                            raw_signals, combineds, valid_confs = run_explainability_batch(
                                adapter=adapter,
                                explainer=explainer,
                                dataset_path=dataset_path,
                                selected_indices=subj_indices,
                                selected_confs=subj_confs,
                                device=device,
                                target_class=target_class,
                                data_format=data_format,
                                index_to_source=_index_to_source if _multi_dataset else None,
                            )

                            adapter.model.float()

                            if len(raw_signals) == 0:
                                print(f"  [Skip] No valid samples for subject {subj_id}")
                                continue

                            n_valid = len(raw_signals)
                            title_prefix = (
                                f"{model_type.upper()} | {task} | class {target_class} | "
                                f"{method.upper()} | Subject: {subj_id} | n={n_valid} segs  conf≥{tp_conf_threshold}"
                            )
                            subtitle = (
                                f"Subject {subj_id}  |  class {target_class}  |  method: {method.upper()}  |  "
                                f"n={n_valid}  conf≥{tp_conf_threshold}"
                            )

                            grand_channel_importance, top_pos, top_neg = compute_grand_channel_importance(
                                combineds=combineds,
                                top_k=top_k,
                            )

                            task_type = config.get('task_type', 'state')
                            os.makedirs(spectral_output_dir, exist_ok=True)
                            faithfulness_result = None
                            temporal_faithfulness_result = None
                            grand_temporal_importance = None

                            if task_type == 'event':
                                # spatial faithfulness (通道级，用于主表)
                                faithfulness_result = spatial_faithfulness(
                                    raw_signals=raw_signals, adapter=adapter,
                                    grand_channel_importance=grand_channel_importance,
                                    channel_names=channel_names, top_k=top_k,
                                    output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                                    save=True, show=False, title_prefix=title_prefix,
                                )

                                single_deletion_result = spatial_single_deletion(
                                    raw_signals=raw_signals,
                                    adapter=adapter,
                                    grand_channel_importance=grand_channel_importance,
                                    channel_names=channel_names,
                                    output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                                    save=True,
                                )
                                if faithfulness_result:
                                    faithfulness_result['spearman_rho'] = single_deletion_result['spearman_rho']
                                    faithfulness_result['spearman_p'] = single_deletion_result['spearman_p']
                                    faithfulness_result['single_deletion_deltas'] = single_deletion_result['deltas']

                                # temporal importance (用于 case study 热力图)
                                grand_temporal_importance = compute_grand_temporal_importance_voting(
                                    combineds=combineds, top_k=top_k, confs=valid_confs,
                                )
                                _model_info = adapter.get_model_info()
                                _patch_size = _model_info.get('patch_size', config.get('patch_size', 200))
                                _patch_stride = _model_info.get('patch_stride', _patch_size)
                                _n_patches = combineds[0].shape[1]

                                temporal_faithfulness_result = temporal_faithfulness(
                                    raw_signals=raw_signals,
                                    adapter=adapter,
                                    grand_temporal_importance=grand_temporal_importance,
                                    n_patches=_n_patches,
                                    patch_size=_patch_size,
                                    patch_stride=_patch_stride,
                                    top_k=top_k,
                                    output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                                    save=True, show=False, title_prefix=title_prefix,
                                )

                                temporal_single_del_result = temporal_single_deletion(
                                    raw_signals=raw_signals,
                                    adapter=adapter,
                                    grand_temporal_importance=grand_temporal_importance,
                                    n_patches=_n_patches,
                                    patch_size=_patch_size,
                                    patch_stride=_patch_stride,
                                    output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                                    save=True,
                                )
                                if temporal_faithfulness_result:
                                    temporal_faithfulness_result['spearman_rho'] = temporal_single_del_result['spearman_rho']
                                    temporal_faithfulness_result['spearman_p'] = temporal_single_del_result['spearman_p']
                                    temporal_faithfulness_result['single_deletion_deltas'] = temporal_single_del_result['deltas']

                                # --- Population grand average waveform + temporal attribution ---
                                try:
                                    _plot_grand_avg_waveform_with_attribution(
                                        raw_signals=raw_signals,
                                        grand_temporal_importance=grand_temporal_importance,
                                        channel_names=channel_names,
                                        n_patches=_n_patches,
                                        patch_size=_patch_size,
                                        title=f"Grand Avg Waveform + Temporal Attribution\n{subtitle}",
                                        output_dir=spectral_output_dir,
                                        save=True, show=False,
                                    )
                                except Exception as e:
                                    print(f"    [Warning] Grand avg waveform plot failed: {e}")

                                # --- Top-3 individual sample waveform + attribution ---
                                try:
                                    _plot_top_sample_waveforms(
                                        raw_signals=raw_signals,
                                        combineds=combineds,
                                        valid_confs=valid_confs,
                                        channel_names=channel_names,
                                        n_top_samples=3,
                                        title_prefix=title_prefix,
                                        output_dir=os.path.join(spectral_output_dir, 'sample_waveforms'),
                                        show=False,
                                    )
                                except Exception as e:
                                    print(f"    [Warning] Sample waveform plots failed: {e}")
                            else:
                                from explainability.spectral_attribution import _plot_grand_avg_topomap
                                _plot_grand_avg_topomap(
                                    grand_channel_importance=grand_channel_importance,
                                    channel_names=channel_names,
                                    top_pos=top_pos,
                                    top_neg=top_neg,
                                    n_samples=n_valid,
                                    output_dir=spectral_output_dir,
                                    subtitle=subtitle,
                                    title_prefix=title_prefix,
                                    save=True, show=False,
                                )

                                faithfulness_result = spatial_faithfulness(
                                    raw_signals=raw_signals, adapter=adapter,
                                    grand_channel_importance=grand_channel_importance,
                                    channel_names=channel_names, top_k=top_k,
                                    output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                                    save=True, show=False, title_prefix=title_prefix,
                                )

                                single_deletion_result = spatial_single_deletion(
                                    raw_signals=raw_signals,
                                    adapter=adapter,
                                    grand_channel_importance=grand_channel_importance,
                                    channel_names=channel_names,
                                    output_dir=os.path.join(spectral_output_dir, 'faithfulness'),
                                    save=True,
                                )
                                if faithfulness_result:
                                    faithfulness_result['spearman_rho'] = single_deletion_result['spearman_rho']
                                    faithfulness_result['spearman_p'] = single_deletion_result['spearman_p']
                                    faithfulness_result['single_deletion_deltas'] = single_deletion_result['deltas']

                            # 缓存该受试者的信号供 PSD
                            if target_class not in _psd_cache:
                                _psd_cache[target_class] = []
                            _psd_cache[target_class].extend(raw_signals)

                            # 频段归因
                            if not skip_band:
                                try:
                                    band_out = os.path.join(spectral_output_dir, 'band_attribution')
                                    os.makedirs(band_out, exist_ok=True)
                                    other_class_signals = []
                                    if _need_donors:
                                        for cls in target_classes:
                                            if cls == target_class:
                                                continue
                                            if cls in _psd_cache:
                                                other_class_signals.extend(_psd_cache[cls])
                                            if cls in _psd_supplement:
                                                other_class_signals.extend(_psd_supplement[cls])
                                    band_result = grand_band_attribution(
                                        raw_signals=raw_signals,
                                        adapter=adapter,
                                        target_class=target_class,
                                        fs=float(config.get('fs', 200)),
                                        methods=band_methods,
                                        baseline_mode=band_baseline,
                                        model_type=model_type,
                                        channel_names=channel_names,
                                        grand_channel_importance=None,
                                        top_k=top_k,
                                        output_dir=band_out,
                                        save=True, show=False,
                                        seed=seed,
                                        n_workers=n_workers,
                                        other_class_signals=other_class_signals if other_class_signals else None,
                                    )
                                except Exception as e:
                                    import traceback
                                    print(f"    [Warning] Band attribution failed: {e}")
                                    traceback.print_exc()

                            # 保存 run_info
                            _write_run_info(
                                spectral_output_dir,
                                Model=model_type, Checkpoint=checkpoint_path or 'N/A',
                                Task=task, Method=method, Dataset=dataset_path,
                                TargetClass=target_class, SampleType='TP',
                                SubjectID=subj_id, NSamples=n_valid,
                                ConfThreshold=tp_conf_threshold,
                                _topk_channels=[(channel_names[i], grand_channel_importance[i]) for i in top_pos],
                            )

                            import json
                            summary_json = {
                                'model': model_type, 'task': task, 'method': method,
                                'target_class': target_class, 'sample_type': 'TP',
                                'subject_id': subj_id,
                                'n_samples': n_valid, 'mean_confidence': mean_conf,
                                'task_type': task_type,
                                'channel_names': channel_names,
                                'grand_channel_importance': grand_channel_importance.tolist(),
                            }
                            if grand_temporal_importance is not None:
                                summary_json['grand_temporal_importance'] = grand_temporal_importance.tolist()
                            if faithfulness_result:
                                summary_json['faithfulness'] = {
                                    'aopc_gain': faithfulness_result.get('aopc_gain'),
                                    'aopc_attribution': faithfulness_result.get('aopc_attribution'),
                                    'aopc_random': faithfulness_result.get('aopc_random'),
                                    'spearman_rho': faithfulness_result.get('spearman_rho'),
                                    'spearman_p': faithfulness_result.get('spearman_p'),
                                    'single_deletion_deltas': faithfulness_result.get('single_deletion_deltas'),
                                }
                            if temporal_faithfulness_result:
                                summary_json['temporal_faithfulness'] = {
                                    'aopc_gain': temporal_faithfulness_result.get('aopc_gain'),
                                    'aopc_attribution': temporal_faithfulness_result.get('aopc_attribution'),
                                    'aopc_random': temporal_faithfulness_result.get('aopc_random'),
                                    'spearman_rho': temporal_faithfulness_result.get('spearman_rho'),
                                    'spearman_p': temporal_faithfulness_result.get('spearman_p'),
                                    'single_deletion_deltas': temporal_faithfulness_result.get('single_deletion_deltas'),
                                }
                            summary_path = os.path.join(spectral_output_dir, 'population_summary.json')
                            with open(summary_path, 'w', encoding='utf-8') as f:
                                json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
                            print(f"  [Saved] {summary_path}")

                        except Exception as e:
                            import traceback
                            print(f"  [Error] method={method}, subject={subj_id}: {e}")
                            traceback.print_exc()
                            continue

        print(f"\n[SUCCESS] Per-subject analyses completed. Results in: {output_dir}")
        return
    # ===================== 结束受试者级分析模式 =====================

    for target_class in target_classes:
        print(f"\n{'='*50}")
        print(f"  Analyzing class {target_class}...")

        if not skip_time:
            for method in methods:
                print(f"\n  Method: {method.upper()}")
                try:
                    model.eval()
                    extra_kwargs = {}
                    if method in ('ig', 'integrated_gradients'):
                        extra_kwargs['n_steps'] = 20
                    explainer = ExplainabilityRegistry.create(
                        method, adapter, device=device, **extra_kwargs
                    )

                    print(f"  --- TP (pred==label=={target_class}) ---")
                    tp_result = _run_one_group(target_class, is_fp=False, method=method, explainer=explainer)
                    tp_raw, tp_top_chs = tp_result
                    if tp_raw is not None and target_class not in raw_signals_by_class:
                        raw_signals_by_class[target_class] = tp_raw

                    if is_binary:
                        print(f"  --- FP (pred=={target_class}, label!={target_class}) ---")
                        _run_one_group(target_class, is_fp=True, method=method, explainer=explainer)

                except Exception as e:
                    import traceback
                    print(f"  [Error] method={method}, class={target_class}: {e}")
                    traceback.print_exc()
                    continue

        # 缓存归因阶段的全部信号供 PSD 使用，然后释放大数据
        if target_class in raw_signals_by_class:
            sigs = raw_signals_by_class[target_class]
            _psd_cache[target_class] = sigs
            del raw_signals_by_class[target_class]

    # 4. PSD 对比分析（所有类别跑完后统一做，始终执行）
    # 固定每类 PSD_N_SAMPLES 个样本：TP缓存优先 + 预缓存补充
    # PSD 保存在与 TP/FP 同级目录下
    if len(target_classes) >= 2:
        try:
            from explainability.spectral_band_attribution import plot_psd_comparison, compute_band_power_context
            ckpt_stem = Path(checkpoint_path).stem if checkpoint_path else 'no_ckpt'
            _psd_method = methods[0] if methods else (band_methods[0] if band_methods else 'default')
            psd_output_dir = os.path.join(
                output_dir, 'population', task,
                f'{model_type}_{ckpt_stem}', _psd_method, 'psd_comparison'
            )
            os.makedirs(psd_output_dir, exist_ok=True)

            psd_signals_by_class = {}

            for cls in target_classes:
                # 优先使用归因阶段缓存的 TP 信号
                cached_tp = _psd_cache.get(cls, [])
                supplement = _psd_supplement.get(cls, [])

                # 合并：TP优先，不足部分用补充样本填充，总数不超过 PSD_N_SAMPLES
                combined = list(cached_tp)
                remaining = PSD_N_SAMPLES - len(combined)
                if remaining > 0 and supplement:
                    combined.extend(supplement[:remaining])

                # 兜底：如果合并后仍不足 PSD_N_SAMPLES，从数据集补读
                if len(combined) < PSD_N_SAMPLES:
                    cls_indices = [r['index'] for r in records
                                   if r.get('label') is not None and int(r['label']) == cls]
                    already_used = set(id(s) for s in combined)
                    rng_psd = random.Random(seed)
                    rng_psd.shuffle(cls_indices)
                    need = PSD_N_SAMPLES - len(combined)
                    loaded_extra = 0
                    for idx in cls_indices:
                        if loaded_extra >= need:
                            break
                        try:
                            if _multi_dataset and idx in _index_to_source:
                                _src_path, _src_idx = _index_to_source[idx]
                                data, _ = load_from_dataset(_src_path, _src_idx, data_format)
                            else:
                                data, _ = load_from_dataset(dataset_path, idx, data_format)
                            if isinstance(data, torch.Tensor):
                                raw = data.cpu().numpy()
                            elif isinstance(data, np.ndarray):
                                raw = data
                            else:
                                continue
                            if raw.ndim == 4:
                                raw = raw.squeeze(0)
                            if raw.ndim == 3:
                                if raw.shape[0] == 1:
                                    raw = raw.squeeze(0)
                                else:
                                    raw = raw.reshape(raw.shape[0], -1)
                            combined.append(raw)
                            loaded_extra += 1
                        except Exception:
                            continue
                    if loaded_extra > 0:
                        print(f"  [PSD] class {cls}: fallback loaded {loaded_extra} extra samples")

                if len(combined) == 0:
                    print(f"  [PSD] class {cls}: no samples available, skipping")
                    continue

                psd_signals_by_class[cls] = combined[:PSD_N_SAMPLES]
                n_tp = min(len(cached_tp), PSD_N_SAMPLES)
                n_sup = len(psd_signals_by_class[cls]) - n_tp
                print(f"  [PSD] class {cls}: {len(psd_signals_by_class[cls])} samples "
                      f"(TP={n_tp}, supplement/fallback={n_sup})")

            class_labels = {c: f"class {c}" for c in psd_signals_by_class}
            top_channels = list(all_top_channels)[:10] if all_top_channels else (channel_names[:5] if channel_names else None)

            print(f"\n{'='*50}")
            print(f"  PSD comparison across {len(psd_signals_by_class)} classes "
                  f"(target: {PSD_N_SAMPLES} samples/class)...")
            for cls, sigs in psd_signals_by_class.items():
                amps = [np.abs(s).max() for s in sigs]
                stds = [s.std() for s in sigs]
                print(f"  [PSD DEBUG] class {cls}: n={len(sigs)}, shape={sigs[0].shape}, "
                      f"amp_median={np.median(amps):.4f}, amp_min={min(amps):.6f}, amp_max={max(amps):.4f}, "
                      f"std_median={np.median(stds):.4f}")
            plot_psd_comparison(
                raw_signals_by_class=psd_signals_by_class,
                fs=float(config.get('fs', 200)),
                channel_names=channel_names,
                top_channels=top_channels,
                class_labels=class_labels,
                output_dir=psd_output_dir,
                save=True,
                show=False,
            )
        except Exception as e:
            import traceback
            print(f"  [Warning] PSD comparison failed: {e}")
            traceback.print_exc()

    print(f"\n[SUCCESS] All analyses completed. Results in: {output_dir}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(description='Population-level EEG Attribution Analysis')

    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-from-dataset', type=str, required=True, nargs='+',
                        help='Path(s) to dataset. Multiple paths or a parent dir with sub-dirs.')
    parser.add_argument('--data-format', type=str, default=None,
                        choices=['lmdb', 'npy', 'npz', 'pt', 'pt_dir'])
    parser.add_argument('--method', type=str, default=None,
                        help='Single explainability method')
    parser.add_argument('--methods', type=str, default=None,
                        help='Multiple methods, comma-separated (e.g. gradcam,ig)')
    parser.add_argument('--all-methods', action='store_true',
                        help='Run all available explainability methods')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Single target class (default: all classes)')
    parser.add_argument('--target-classes', type=str, default=None,
                        help='Multiple classes, comma-separated (e.g. 0,1)')
    parser.add_argument('--tp-conf-threshold', type=float, default=0.7,
                        help='Confidence threshold for TP samples (default: 0.7)')
    parser.add_argument('--fp-conf-threshold', type=float, default=0.8,
                        help='Confidence threshold for FP samples (default: 0.8)')
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='Number of samples per class for both time and band attribution (-1 = all, default: 40)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-K channels for spectral attribution (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./population_results_totally_1')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)

    # Report / LLM removed — use `python -m explainability.llm_interpret_population` separately

    # Band attribution
    parser.add_argument('--band-methods', type=str, default=None,
                        help='Band attribution methods, comma-separated (e.g. occlusion,shap). If not set, skip band attribution.')
    parser.add_argument('--band-baseline', type=str, default='auto',
                        choices=['zero', 'gaussian', 'phase_shuffle', 'class_permute', 'auto'],
                        help='Band attribution baseline mode (default: auto → class_permute if multi-class, else zero)')
    parser.add_argument('--skip-band-attribution', action='store_true',
                        help='Skip band attribution analysis')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of threads for band attribution (default: 1)')

    # Subject-level analysis
    parser.add_argument('--per-subject', action='store_true',
                        help='Per-subject analysis: confidence filter → group by subject → analyze top subjects')
    parser.add_argument('--n-top-subjects', type=int, default=1,
                        help='Number of top subjects per class to analyze (default: 1)')
    parser.add_argument('--subject-ids', type=str, default=None,
                        help='Subject IDs to use. Two formats supported: '
                             'simple "008,020" (all classes) or '
                             'per-class "0:008,020;2:015,030" (auto-sets target classes). '
                             'Overrides automatic top-subject selection. Implies --per-subject.')

    args = parser.parse_args()

    # --subject-ids implies --per-subject
    if args.subject_ids:
        args.per_subject = True

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    # 确定方法列表
    if args.all_methods:
        methods = ExplainabilityRegistry.list_methods()
    elif args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
    elif args.method:
        methods = [args.method]
    else:
        methods = None  # 不跑时间维度归因
    print(f"Methods: {methods}")

    # 确定类别列表（None = 自动检测所有类别）
    if args.target_class is not None:
        target_classes = [args.target_class]
    elif args.target_classes:
        target_classes = [int(c) for c in args.target_classes.split(',')]
    else:
        target_classes = None  # 自动检测

    # 如果 --subject-ids 使用按类格式且未指定 target_classes，自动从中推断
    if target_classes is None and args.subject_ids and ':' in args.subject_ids:
        parsed_sids = _parse_subject_ids(args.subject_ids)
        if isinstance(parsed_sids, dict):
            target_classes = sorted(parsed_sids.keys())
            print(f"  [Info] Auto-detected target_classes from --subject-ids: {target_classes}")

    # 加载 yaml config
    config_file = os.path.join(project_root, 'configs', f'{args.model_type}.yaml')
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    if args.task not in yaml_data.get('CONFIGS', {}):
        available = list(yaml_data.get('CONFIGS', {}).keys())
        raise ValueError(f"Task '{args.task}' not found. Available: {available}")

    config = yaml_data['CONFIGS'][args.task]
    # 注入 task_type（从 TaskConfig 获取，YAML 中未定义时自动补充）
    if 'task_type' not in config:
        from explainability.task_configs import TASK_CONFIGS
        if args.task.lower() in TASK_CONFIGS:
            config['task_type'] = TASK_CONFIGS[args.task.lower()].task_type
    print(f"\nConfig for task '{args.task}':")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 解析数据路径：支持多路径或父目录
    data_paths = args.data_from_dataset  # list
    resolved_paths = []
    for dp in data_paths:
        p = Path(dp)
        if p.is_dir():
            # 如果目录本身是数据集（含 data.mdb / pkl / pt 等），直接用
            has_data = ((p / 'data.mdb').exists()
                        or list(p.glob('*.pkl'))
                        or list(p.glob('*.pt'))
                        or list(p.glob('*.npy'))
                        or list(p.glob('*.mat')))
            if has_data:
                resolved_paths.append(str(p))
            else:
                # 否则扫描子目录
                subdirs = sorted(d for d in p.iterdir() if d.is_dir())
                for sd in subdirs:
                    sd_has_data = ((sd / 'data.mdb').exists()
                                   or list(sd.glob('*.pkl'))
                                   or list(sd.glob('*.pt'))
                                   or list(sd.glob('*.npy'))
                                   or list(sd.glob('*.mat')))
                    if sd_has_data:
                        resolved_paths.append(str(sd))
                if not resolved_paths:
                    resolved_paths.append(str(p))
        else:
            resolved_paths.append(str(p))

    if len(resolved_paths) > 1:
        print(f"\n  Resolved {len(resolved_paths)} dataset paths:")
        for rp in resolved_paths:
            print(f"    {rp}")

    dataset_path = resolved_paths[0] if len(resolved_paths) == 1 else resolved_paths

    run_population_analysis(
        model_type=args.model_type,
        config=config,
        methods=methods,
        dataset_path=dataset_path,
        checkpoint_path=args.checkpoint,
        task=args.task,
        device=args.device,
        target_classes=target_classes,
        tp_conf_threshold=args.tp_conf_threshold,
        fp_conf_threshold=args.fp_conf_threshold,
        n_samples=args.n_samples,
        top_k=args.top_k,
        output_dir=args.output_dir,
        data_format=args.data_format,
        seed=args.seed,
        band_methods=[m.strip() for m in args.band_methods.split(',')] if args.band_methods else None,
        band_baseline=args.band_baseline,
        skip_band_attribution=args.skip_band_attribution,
        n_workers=args.n_workers,
        per_subject=args.per_subject,
        n_top_subjects=args.n_top_subjects,
        subject_ids=_parse_subject_ids(args.subject_ids) if args.subject_ids else None,
        classifier_type=config.get('classifier_type', 'all_patch_reps'),
        d_model=config.get('d_model', 200),
    )

 
if __name__ == '__main__':
    main()
