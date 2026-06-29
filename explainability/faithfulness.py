"""
忠实度分析 (Faithfulness Analysis)
Spatial Faithfulness for EEG Attribution

基于群体归因得到的通道重要度排序，在 TP/FP 样本上做累积遮蔽，
评估模型输出置信度下降曲线和 AOPC 指标。

AOPC 采用标准定义 (Samek et al. 2017):
  AOPC = (1/K) * sum_{k=1}^{K} (f(x) - f(x_{mask_k}))
即 baseline logit 与逐步遮蔽后 logit 的平均差值，无 per-sample normalization。

遮蔽方式: 用该批样本的通道均值替换（避免 zero-out 的 OOD 问题）
对比基线: 随机顺序遮蔽（证明归因比随机好）
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import random
from typing import List, Dict, Optional
from scipy.stats import spearmanr


# ==================== 核心推理函数 ====================

def _get_pred(output: torch.Tensor, is_binary: bool, threshold: float = 0.5) -> int:
    if is_binary:
        return int(torch.sigmoid(output.float()).mean().item() > threshold)
    else:
        return int(output.argmax(dim=-1).item())


def _get_confidence(output: torch.Tensor, pred_class: int, is_binary: bool) -> float:
    if is_binary:
        logit = output.float().mean().item()
        return logit if pred_class == 1 else -logit
    else:
        return output.float()[0, pred_class].item()


def _infer_batch(
    raw_signals: List[np.ndarray],
    pred_classes: List[int],
    adapter,
    mask_channel_indices: List[int],
    channel_means: np.ndarray,
    is_binary: bool,
    batch_size: int = 32,
) -> float:
    total_conf = 0.0
    n_total = 0

    for start in range(0, len(raw_signals), batch_size):
        batch_raws = raw_signals[start:start + batch_size]
        batch_preds = pred_classes[start:start + batch_size]
        tensors = []
        for raw in batch_raws:
            sig = raw.copy()
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig.squeeze(0)
            elif sig.ndim == 3:
                sig = sig.reshape(sig.shape[0], -1)
            for ch_idx in mask_channel_indices:
                sig[ch_idx] = channel_means[ch_idx]
            tensors.append(adapter.prepare_input(sig))

        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = adapter.forward(batch)

        for i, pred_class in enumerate(batch_preds):
            total_conf += _get_confidence(output[i:i+1], pred_class, is_binary)
            n_total += 1

    return float(total_conf / n_total) if n_total > 0 else 0.0


# ==================== 主函数 ====================

def spatial_faithfulness(
    raw_signals: List[np.ndarray],
    adapter,
    grand_channel_importance: np.ndarray,
    channel_names: List[str],
    top_k: int = 5,
    batch_size: int = 32,
    n_random_seeds: int = 10,
    output_dir: str = './faithfulness',
    save: bool = True,
    show: bool = False,
    title_prefix: str = '',
) -> Dict:
    """
    空间忠实度分析：累积遮蔽正向 Top-K 通道，观察预测置信度下降。
    AOPC 使用标准定义：baseline logit 与遮蔽后 logit 的平均差值。
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(raw_signals)
    is_binary = getattr(adapter, 'is_binary', False)
    threshold = 0.6394 if getattr(adapter, 'model_name', None) == 'biot' else 0.5
    print(f"\n[Faithfulness] n_samples={n_samples}, top_k={top_k}, is_binary={is_binary}")

    channel_means = _compute_channel_means(raw_signals)

    attribution_order = np.argsort(grand_channel_importance)[::-1][:top_k].tolist()
    channel_order = [channel_names[i] for i in attribution_order]
    actual_k = len(attribution_order)
    print(f"  Attribution mask order (top-{actual_k}): {channel_order}")

    # baseline
    t0 = time.time()
    pred_classes = []
    for start in range(0, n_samples, batch_size):
        batch_raws = raw_signals[start:start + batch_size]
        tensors = []
        for raw in batch_raws:
            sig = raw.copy()
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig.squeeze(0)
            elif sig.ndim == 3:
                sig = sig.reshape(sig.shape[0], -1)
            tensors.append(adapter.prepare_input(sig))
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = adapter.forward(batch)
        for i in range(len(batch_raws)):
            pred_classes.append(_get_pred(output[i:i+1], is_binary, threshold))

    conf_base = _infer_batch(raw_signals, pred_classes, adapter, [], channel_means, is_binary, batch_size)
    print(f"  baseline conf={conf_base:.4f}  ({time.time()-t0:.1f}s)")

    conf_all_masked = _infer_batch(raw_signals, pred_classes, adapter,
                                   list(range(len(channel_names))), channel_means, is_binary, batch_size)
    print(f"  [DEBUG] all channels masked logit={conf_all_masked:.4f}, baseline logit={conf_base:.4f}")

    # 归因顺序累积遮蔽
    attribution_curve = [conf_base]
    cumulative_mask = []
    for step, ch_idx in enumerate(attribution_order):
        cumulative_mask.append(ch_idx)
        t0 = time.time()
        conf = _infer_batch(raw_signals, pred_classes, adapter, cumulative_mask, channel_means, is_binary, batch_size)
        attribution_curve.append(conf)
        print(f"  Step {step+1}/{actual_k} mask={[channel_names[i] for i in cumulative_mask]} "
              f"logit={conf:.4f}  ({time.time()-t0:.1f}s)")

    # 随机顺序基线
    all_indices = list(range(len(channel_names)))
    random_curves = []
    random_orders = []
    for seed in range(n_random_seeds):
        random.seed(seed)
        rand_order = random.sample(all_indices, actual_k)
        random_orders.append([channel_names[i] for i in rand_order])
        rand_curve = [conf_base]
        cumulative_mask = []
        for ch_idx in rand_order:
            cumulative_mask.append(ch_idx)
            conf = _infer_batch(raw_signals, pred_classes, adapter, cumulative_mask, channel_means, is_binary, batch_size)
            rand_curve.append(conf)
        random_curves.append(rand_curve)
        print(f"  Random seed={seed} order={random_orders[-1]} "
              f"final_logit={rand_curve[-1]:.4f}")

    random_curve = np.mean(random_curves, axis=0).tolist()

    # 标准 AOPC：直接用 raw logit drop，不做 per-sample normalization
    aopc_attr = _compute_aopc(conf_base, attribution_curve[1:])
    aopc_rand = _compute_aopc(conf_base, random_curve[1:])
    print(f"  AOPC attribution={aopc_attr:.4f}, random={aopc_rand:.4f}, "
          f"gain={aopc_attr - aopc_rand:.4f}")

    # 画图 + 报告（用 raw logit curves）
    _plot_faithfulness(
        attribution_curve=attribution_curve,
        random_curve=random_curve,
        channel_order=channel_order,
        aopc_attr=aopc_attr,
        aopc_rand=aopc_rand,
        output_dir=output_dir,
        save=save,
        show=show,
        title_prefix=title_prefix,
    )
    _save_report(
        attribution_curve=attribution_curve,
        random_curve=random_curve,
        channel_order=channel_order,
        random_orders=random_orders,
        aopc_attr=aopc_attr,
        aopc_rand=aopc_rand,
        n_samples=n_samples,
        output_dir=output_dir,
    )

    return {
        'attribution_curve': attribution_curve,
        'random_curve': random_curve,
        'random_curves': [list(c) for c in random_curves],
        'aopc_attribution': aopc_attr,
        'aopc_random': aopc_rand,
        'aopc_gain': aopc_attr - aopc_rand,
        'channel_order': channel_order,
        'random_orders': random_orders,
        'baseline_logit': conf_base,
        'all_masked_logit': conf_all_masked,
    }


# ==================== 辅助函数 ====================

def _compute_channel_means(raw_signals: List[np.ndarray]) -> np.ndarray:
    stack = []
    for raw in raw_signals:
        sig = raw.copy()
        if sig.ndim == 3 and sig.shape[0] == 1:
            sig = sig.squeeze(0)
        elif sig.ndim == 3:
            sig = sig.reshape(sig.shape[0], -1)
        stack.append(sig)
    return np.mean(stack, axis=0).astype(np.float32)


def _compute_aopc(baseline: float, curve: List[float]) -> float:
    """标准 AOPC = mean(baseline - curve[k])，值越大归因越忠实"""
    if not curve:
        return 0.0
    return float(np.mean([baseline - v for v in curve]))


# ==================== 可视化 ====================

def _plot_faithfulness(
    attribution_curve: List[float],
    random_curve: List[float],
    channel_order: List[str],
    aopc_attr: float,
    aopc_rand: float,
    output_dir: str,
    save: bool,
    show: bool,
    title_prefix: str = '',
):
    steps = list(range(len(attribution_curve)))
    xtick_labels = ['baseline'] + channel_order

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(steps, attribution_curve, 'o-', color='#C44E52', linewidth=1.8,
            markersize=5, label=f'Attribution (AOPC={aopc_attr:.3f})')
    ax.plot(steps, random_curve, 's--', color='#4C72B0', linewidth=1.4,
            markersize=4.5, alpha=0.7, label=f'Random (AOPC={aopc_rand:.3f})')

    ax.set_xticks(steps)
    ax.set_xticklabels(xtick_labels, fontsize=11, rotation=20, ha='right')
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel('Masked channels (cumulative)', fontsize=12)
    ax.set_ylabel('Mean Logit of Predicted Class', fontsize=12)
    ax.set_title('Spatial Faithfulness', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.85)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    for x, y in zip(steps, attribution_curve):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=10, color='#C44E52')

    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, 'spatial_faithfulness.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"[Saved] {path} (+pdf)")
    if show:
        plt.show()
    plt.close()


def _save_report(
    attribution_curve: List[float],
    random_curve: List[float],
    channel_order: List[str],
    random_orders: List[List[str]],
    aopc_attr: float,
    aopc_rand: float,
    n_samples: int,
    output_dir: str,
):
    path = os.path.join(output_dir, 'faithfulness_report.txt')
    lines = [
        "=" * 50,
        "Spatial Faithfulness Report (Standard AOPC)",
        "=" * 50,
        f"Samples evaluated : {n_samples}",
        f"Mask order        : {channel_order}",
        "",
        f"{'Step':<6} {'Channel':<12} {'Logit':>12} {'Random':>10}",
        "-" * 44,
    ]
    for i, (a, r) in enumerate(zip(attribution_curve, random_curve)):
        ch = 'baseline' if i == 0 else channel_order[i - 1]
        lines.append(f"{i:<6} {ch:<12} {a:>12.4f} {r:>10.4f}")
    rand_section = ["", "Random baseline channel orders (per seed):"]
    for seed_idx, order in enumerate(random_orders):
        rand_section.append(f"  Seed {seed_idx}: {order}")
    lines += rand_section + [
        "",
        "-" * 44,
        f"AOPC (attribution) : {aopc_attr:.4f}",
        f"AOPC (random)      : {aopc_rand:.4f}",
        f"AOPC gain          : {aopc_attr - aopc_rand:.4f}",
        "=" * 50,
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Saved] {path}")


# ==================== Temporal Faithfulness ====================

def _infer_batch_temporal(
    raw_signals: List[np.ndarray],
    pred_classes: List[int],
    adapter,
    mask_patch_indices: List[int],
    n_patches: int,
    patch_size: int,
    is_binary: bool,
    batch_size: int = 32,
    rng_seed: int = 42,
    patch_stride: Optional[int] = None,
) -> float:
    """对一批样本做 patch 遮蔽后推理，返回预测类别 logit 均值。
    遮蔽方式：用同均值/标准差的高斯噪声替换，破坏时间结构但保持振幅统计。
    """
    if patch_stride is None:
        patch_stride = patch_size
    rng = np.random.RandomState(rng_seed)
    total_conf = 0.0
    n_total = 0

    for start in range(0, len(raw_signals), batch_size):
        batch_raws = raw_signals[start:start + batch_size]
        batch_preds = pred_classes[start:start + batch_size]
        tensors = []
        for raw in batch_raws:
            sig = raw.copy()
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig.squeeze(0)
            elif sig.ndim == 3:
                sig = sig.reshape(sig.shape[0], -1)
            for p_idx in mask_patch_indices:
                start_t = p_idx * patch_stride
                end_t = start_t + patch_size
                if end_t <= sig.shape[1]:
                    patch = sig[:, start_t:end_t]
                    mu = patch.mean(axis=1, keepdims=True)
                    std = patch.std(axis=1, keepdims=True) + 1e-8
                    sig[:, start_t:end_t] = rng.randn(*patch.shape) * std + mu
            tensors.append(adapter.prepare_input(sig))

        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = adapter.forward(batch)

        for i, pred_class in enumerate(batch_preds):
            total_conf += _get_confidence(output[i:i+1], pred_class, is_binary)
            n_total += 1

    return float(total_conf / n_total) if n_total > 0 else 0.0


def temporal_faithfulness(
    raw_signals: List[np.ndarray],
    adapter,
    grand_temporal_importance: np.ndarray,
    n_patches: int,
    patch_size: int,
    patch_stride: Optional[int] = None,
    top_k: int = 5,
    batch_size: int = 32,
    n_random_seeds: int = 10,
    output_dir: str = './faithfulness',
    save: bool = True,
    show: bool = False,
    title_prefix: str = '',
) -> Dict:
    """
    时间忠实度分析：累积遮蔽正向 Top-K patches，观察预测 logit 下降。
    AOPC 使用标准定义：baseline logit 与遮蔽后 logit 的平均差值，无 per-sample normalization。
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(raw_signals)
    is_binary = getattr(adapter, 'is_binary', False)
    threshold = 0.6394 if getattr(adapter, 'model_name', None) == 'biot' else 0.5
    print(f"\n[Temporal Faithfulness] n_samples={n_samples}, top_k={top_k}, "
          f"n_patches={n_patches}, patch_size={patch_size}, is_binary={is_binary}")

    actual_k = min(top_k, n_patches)
    attribution_order = np.argsort(grand_temporal_importance)[::-1][:actual_k].tolist()
    patch_order = [f'P{i}' for i in attribution_order]
    print(f"  Attribution mask order (top-{actual_k}): {patch_order}")

    # baseline
    t0 = time.time()
    pred_classes = []
    for start_idx in range(0, n_samples, batch_size):
        batch_raws = raw_signals[start_idx:start_idx + batch_size]
        tensors = []
        for raw in batch_raws:
            sig = raw.copy()
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig.squeeze(0)
            elif sig.ndim == 3:
                sig = sig.reshape(sig.shape[0], -1)
            tensors.append(adapter.prepare_input(sig))
        batch_t = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = adapter.forward(batch_t)
        for i in range(len(batch_raws)):
            pred_classes.append(_get_pred(output[i:i+1], is_binary, threshold))

    _stride = patch_stride if patch_stride is not None else patch_size
    conf_base = _infer_batch_temporal(
        raw_signals, pred_classes, adapter, [],
        n_patches, patch_size, is_binary, batch_size, patch_stride=_stride)
    print(f"  baseline logit={conf_base:.4f}  ({time.time()-t0:.1f}s)")

    conf_all_masked = _infer_batch_temporal(
        raw_signals, pred_classes, adapter, list(range(n_patches)),
        n_patches, patch_size, is_binary, batch_size, patch_stride=_stride)
    print(f"  [DEBUG] all patches masked logit={conf_all_masked:.4f}")

    # 归因顺序累积遮蔽
    attribution_curve = [conf_base]
    cumulative_mask = []
    for step, p_idx in enumerate(attribution_order):
        cumulative_mask.append(p_idx)
        t0 = time.time()
        conf = _infer_batch_temporal(
            raw_signals, pred_classes, adapter, cumulative_mask,
            n_patches, patch_size, is_binary, batch_size, patch_stride=_stride)
        attribution_curve.append(conf)
        print(f"  Step {step+1}/{actual_k} mask=[{','.join(f'P{i}' for i in cumulative_mask)}] "
              f"logit={conf:.4f}  ({time.time()-t0:.1f}s)")

    # 随机顺序基线
    all_patch_indices = list(range(n_patches))
    random_curves = []
    random_orders = []

    for seed in range(n_random_seeds):
        random.seed(seed)
        rand_order = random.sample(all_patch_indices, actual_k)
        random_orders.append([f'P{i}' for i in rand_order])
        rand_curve = [conf_base]
        cumulative_mask = []
        for p_idx in rand_order:
            cumulative_mask.append(p_idx)
            conf = _infer_batch_temporal(
                raw_signals, pred_classes, adapter, cumulative_mask,
                n_patches, patch_size, is_binary, batch_size, patch_stride=_stride)
            rand_curve.append(conf)
        random_curves.append(rand_curve)
        print(f"  Random seed={seed} final_logit={rand_curve[-1]:.4f}")

    random_curve = np.mean(random_curves, axis=0).tolist()

    # 标准 AOPC：直接用 raw logit drop，不做 per-sample normalization
    aopc_attr = _compute_aopc(conf_base, attribution_curve[1:])
    aopc_rand = _compute_aopc(conf_base, random_curve[1:])
    print(f"  AOPC attribution={aopc_attr:.4f}, random={aopc_rand:.4f}, "
          f"gain={aopc_attr - aopc_rand:.4f}")

    _plot_temporal_faithfulness(
        attribution_curve=attribution_curve,
        random_curve=random_curve,
        patch_order=patch_order,
        aopc_attr=aopc_attr,
        aopc_rand=aopc_rand,
        output_dir=output_dir,
        save=save,
        show=show,
        title_prefix=title_prefix,
    )
    _save_temporal_report(
        attribution_curve=attribution_curve,
        random_curve=random_curve,
        patch_order=patch_order,
        random_orders=random_orders,
        aopc_attr=aopc_attr,
        aopc_rand=aopc_rand,
        n_samples=n_samples,
        output_dir=output_dir,
    )

    return {
        'attribution_curve': attribution_curve,
        'random_curve': random_curve,
        'random_curves': [list(c) for c in random_curves],
        'aopc_attribution': aopc_attr,
        'aopc_random': aopc_rand,
        'aopc_gain': aopc_attr - aopc_rand,
        'patch_order': patch_order,
        'random_orders': random_orders,
        'baseline_logit': conf_base,
        'all_masked_logit': conf_all_masked,
    }


def _plot_temporal_faithfulness(
    attribution_curve, random_curve, patch_order,
    aopc_attr, aopc_rand, output_dir, save, show, title_prefix='',
):
    steps = list(range(len(attribution_curve)))
    xtick_labels = ['baseline'] + patch_order

    fig, ax = plt.subplots(figsize=(max(7, len(steps) * 1.2), 5))
    ax.plot(steps, attribution_curve, 'o-', color='#C44E52', linewidth=2,
            markersize=6, label=f'Attribution order (AOPC={aopc_attr:.4f})')
    ax.plot(steps, random_curve, 's--', color='#4C72B0', linewidth=1.5,
            markersize=5, alpha=0.7, label=f'Random order (AOPC={aopc_rand:.4f})')

    ax.set_xticks(steps)
    ax.set_xticklabels(xtick_labels, fontsize=9, rotation=15, ha='right')
    ax.set_xlabel('Masked patches (cumulative)', fontsize=11)
    ax.set_ylabel('Mean Logit', fontsize=11)
    ax.set_title((f"{title_prefix}\n" if title_prefix else "") +
                 'Temporal Faithfulness: Logit Drop under Patch Masking', fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    for x, y in zip(steps, attribution_curve):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7.5, color='#C44E52')

    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, 'temporal_faithfulness.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


def _save_temporal_report(
    attribution_curve, random_curve, patch_order, random_orders,
    aopc_attr, aopc_rand, n_samples, output_dir,
):
    path = os.path.join(output_dir, 'temporal_faithfulness_report.txt')
    lines = [
        "=" * 50,
        "Temporal Faithfulness Report (Standard AOPC)",
        "=" * 50,
        f"Samples evaluated : {n_samples}",
        f"Mask order        : {patch_order}",
        "",
        f"{'Step':<6} {'Patch':<12} {'Logit':>12} {'Random':>10}",
        "-" * 44,
    ]
    for i, (a, r) in enumerate(zip(attribution_curve, random_curve)):
        p = 'baseline' if i == 0 else patch_order[i - 1]
        lines.append(f"{i:<6} {p:<12} {a:>12.4f} {r:>10.4f}")
    rand_section = ["", "Random baseline patch orders (per seed):"]
    for seed_idx, order in enumerate(random_orders):
        rand_section.append(f"  Seed {seed_idx}: {order}")
    lines += rand_section + [
        "",
        "-" * 44,
        f"AOPC (attribution) : {aopc_attr:.4f}",
        f"AOPC (random)      : {aopc_rand:.4f}",
        f"AOPC gain          : {aopc_attr - aopc_rand:.4f}",
        "=" * 50,
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Saved] {path}")


# ==================== Single-Deletion Faithfulness ====================


def spatial_single_deletion(
    raw_signals: List[np.ndarray],
    adapter,
    grand_channel_importance: np.ndarray,
    channel_names: List[str],
    batch_size: int = 32,
    output_dir: str = './faithfulness',
    save: bool = True,
) -> Dict:
    """
    空间 single-deletion 忠实度：对每个通道独立遮蔽，计算 Spearman ρ。
    """
    os.makedirs(output_dir, exist_ok=True)
    n_channels = len(channel_names)
    is_binary = getattr(adapter, 'is_binary', False)
    threshold = 0.6394 if getattr(adapter, 'model_name', None) == 'biot' else 0.5
    print(f"\n[Spatial Single-Deletion] n_samples={len(raw_signals)}, n_channels={n_channels}")

    channel_means = _compute_channel_means(raw_signals)

    pred_classes = _get_all_pred_classes(raw_signals, adapter, is_binary, batch_size, threshold)
    conf_base = _infer_batch(raw_signals, pred_classes, adapter, [], channel_means, is_binary, batch_size)
    print(f"  baseline logit={conf_base:.4f}")

    deltas = np.zeros(n_channels)
    for ch_idx in range(n_channels):
        conf = _infer_batch(raw_signals, pred_classes, adapter, [ch_idx], channel_means, is_binary, batch_size)
        deltas[ch_idx] = conf_base - conf

    rho, p_value = spearmanr(grand_channel_importance, deltas)
    print(f"  Spearman rho={rho:.4f}, p={p_value:.4f}")

    top_idx = np.argsort(deltas)[::-1][:5]
    print(f"  Top-5 by actual drop:")
    for i in top_idx:
        print(f"    {channel_names[i]}: delta={deltas[i]:.4f}, attribution={grand_channel_importance[i]:.4f}")

    result = {
        'deltas': deltas.tolist(),
        'attribution_scores': grand_channel_importance.tolist(),
        'spearman_rho': float(rho),
        'spearman_p': float(p_value),
        'baseline_logit': conf_base,
        'channel_names': channel_names,
        'n_samples': len(raw_signals),
    }

    if save:
        _save_single_deletion_report(
            deltas=deltas, attribution_scores=grand_channel_importance,
            feature_names=channel_names, rho=rho, p_value=p_value,
            baseline_logit=conf_base, n_samples=len(raw_signals),
            output_dir=output_dir, prefix='spatial',
        )

    return result


def temporal_single_deletion(
    raw_signals: List[np.ndarray],
    adapter,
    grand_temporal_importance: np.ndarray,
    n_patches: int,
    patch_size: int,
    patch_stride: Optional[int] = None,
    batch_size: int = 32,
    output_dir: str = './faithfulness',
    save: bool = True,
) -> Dict:
    """
    时间 single-deletion 忠实度：对每个 patch 独立遮蔽，计算 Spearman ρ。
    """
    os.makedirs(output_dir, exist_ok=True)
    is_binary = getattr(adapter, 'is_binary', False)
    threshold = 0.6394 if getattr(adapter, 'model_name', None) == 'biot' else 0.5
    print(f"\n[Temporal Single-Deletion] n_samples={len(raw_signals)}, n_patches={n_patches}")

    _stride = patch_stride if patch_stride is not None else patch_size
    pred_classes = _get_all_pred_classes(raw_signals, adapter, is_binary, batch_size, threshold)
    conf_base = _infer_batch_temporal(
        raw_signals, pred_classes, adapter, [],
        n_patches, patch_size, is_binary, batch_size, patch_stride=_stride)
    print(f"  baseline logit={conf_base:.4f}")

    deltas = np.zeros(n_patches)
    for p_idx in range(n_patches):
        conf = _infer_batch_temporal(
            raw_signals, pred_classes, adapter, [p_idx],
            n_patches, patch_size, is_binary, batch_size, patch_stride=_stride)
        deltas[p_idx] = conf_base - conf

    patch_names = [f'P{i}' for i in range(n_patches)]

    rho, p_value = spearmanr(grand_temporal_importance[:n_patches], deltas)
    print(f"  Spearman rho={rho:.4f}, p={p_value:.4f}")

    top_idx = np.argsort(deltas)[::-1][:min(5, n_patches)]
    print(f"  Top patches by actual drop:")
    for i in top_idx:
        print(f"    P{i}: delta={deltas[i]:.4f}, attribution={grand_temporal_importance[i]:.4f}")

    result = {
        'deltas': deltas.tolist(),
        'attribution_scores': grand_temporal_importance[:n_patches].tolist(),
        'spearman_rho': float(rho),
        'spearman_p': float(p_value),
        'baseline_logit': conf_base,
        'patch_names': patch_names,
        'n_samples': len(raw_signals),
    }

    if save:
        _save_single_deletion_report(
            deltas=deltas, attribution_scores=grand_temporal_importance[:n_patches],
            feature_names=patch_names, rho=rho, p_value=p_value,
            baseline_logit=conf_base, n_samples=len(raw_signals),
            output_dir=output_dir, prefix='temporal',
        )

    return result


def _get_all_pred_classes(
    raw_signals: List[np.ndarray], adapter, is_binary: bool, batch_size: int,
    threshold: float = 0.5,
) -> List[int]:
    pred_classes = []
    for start in range(0, len(raw_signals), batch_size):
        batch_raws = raw_signals[start:start + batch_size]
        tensors = []
        for raw in batch_raws:
            sig = raw.copy()
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig.squeeze(0)
            elif sig.ndim == 3:
                sig = sig.reshape(sig.shape[0], -1)
            tensors.append(adapter.prepare_input(sig))
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = adapter.forward(batch)
        for i in range(len(batch_raws)):
            pred_classes.append(_get_pred(output[i:i+1], is_binary, threshold))
    return pred_classes


def _save_single_deletion_report(
    deltas: np.ndarray,
    attribution_scores: np.ndarray,
    feature_names: List[str],
    rho: float,
    p_value: float,
    baseline_logit: float,
    n_samples: int,
    output_dir: str,
    prefix: str,
):
    path = os.path.join(output_dir, f'{prefix}_single_deletion_report.txt')
    lines = [
        "=" * 55,
        f"{prefix.capitalize()} Single-Deletion Faithfulness Report",
        "=" * 55,
        f"Samples evaluated : {n_samples}",
        f"Baseline logit    : {baseline_logit:.4f}",
        f"Spearman rho      : {rho:.4f}",
        f"p-value           : {p_value:.4f}",
        "",
        f"{'Feature':<12} {'Attribution':>12} {'Delta':>10} {'Rank_attr':>10} {'Rank_drop':>10}",
        "-" * 55,
    ]

    attr_rank = np.argsort(np.argsort(-attribution_scores)) + 1
    drop_rank = np.argsort(np.argsort(-deltas)) + 1

    for i, name in enumerate(feature_names):
        lines.append(
            f"{name:<12} {attribution_scores[i]:>12.4f} {deltas[i]:>10.4f} "
            f"{attr_rank[i]:>10} {drop_rank[i]:>10}"
        )

    lines += ["", "=" * 55]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Saved] {path}") 