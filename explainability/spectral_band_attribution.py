"""
频段归因模块 — Band-level Attribution for EEG Models

对每个 (channel, band) 组合施加扰动，测量模型输出变化，
得到 (n_channels, n_bands) 的归因矩阵。

支持 5 种方法: occlusion, shap, ig, lime, gradient_shap
支持 3 种基线: zero, gaussian, phase_shuffle
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import wilcoxon, spearmanr

# 复用已有的区域/侧化映射
from explainability.spectral_attribution import (
    REGION_MAP, LATERALITY_LEFT, LATERALITY_RIGHT,
    _ch_region, _ch_laterality,
)

DEFAULT_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# 含 FFT/频谱分支的模型：phase_shuffle 只破坏 time_emb，spectral_emb 不受影响，
# 因此应使用 gaussian 基线（同时破坏时域和频域信息）
FFT_MODELS = {'cbramod', 'eegmamba', 'biot'}

# 纯时域模型：只看时间波形，phase_shuffle 能精确测量"模型依赖该频段时间模式的程度"
TEMPORAL_MODELS = {'labram', 'neurolm', 'eegpt'}


def resolve_band_baseline(baseline_mode: str, model_type: Optional[str] = None,
                           other_class_signals: List[np.ndarray] = None) -> str:
    """将 'auto' 基线模式解析为具体方法。

    - 有 other_class_signals → 'class_permute'（最优，无 1/f 偏差，扰动在训练分布内）
    - 无 other_class_signals + FFT 模型 → 'zero'（bandstop 滤波）
    - 无 other_class_signals + 纯时域模型 → 'phase_shuffle'
    - 未知 → 'zero'
    """
    if baseline_mode != 'auto':
        return baseline_mode
    if other_class_signals is not None and len(other_class_signals) > 0:
        return 'class_permute'
    if model_type is None:
        return 'zero'
    model_lower = model_type.lower()
    if model_lower in TEMPORAL_MODELS:
        return 'phase_shuffle'
    return 'zero'


# ==================== 滤波基线函数 ====================

def _bandstop_filter(signal_1d: np.ndarray, fs: float,
                     f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    lo = max(f_lo / nyq, 1e-5)
    hi = min(f_hi / nyq, 0.9999)
    if lo >= hi:
        return signal_1d.copy()
    sos = butter(order, [lo, hi], btype='bandstop', output='sos')
    return sosfiltfilt(sos, signal_1d).astype(signal_1d.dtype)


def _bandpass_filter(signal_1d: np.ndarray, fs: float,
                     f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    lo = max(f_lo / nyq, 1e-5)
    hi = min(f_hi / nyq, 0.9999)
    if lo >= hi:
        return signal_1d.copy()
    sos = butter(order, [lo, hi], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal_1d).astype(signal_1d.dtype)


def _phase_shuffle_band(signal_1d: np.ndarray, fs: float,
                        f_lo: float, f_hi: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    band_component = _bandpass_filter(signal_1d, fs, f_lo, f_hi)
    fft_vals = np.fft.rfft(band_component)
    phases = np.angle(fft_vals)
    shuffled_phases = rng.permutation(phases)
    fft_shuffled = np.abs(fft_vals) * np.exp(1j * shuffled_phases)
    shuffled_component = np.fft.irfft(fft_shuffled, n=len(signal_1d))
    residual = signal_1d - band_component
    return (residual + shuffled_component).astype(signal_1d.dtype)


def _class_permute_band(signal_1d: np.ndarray, fs: float,
                        f_lo: float, f_hi: float,
                        other_class_signals: List[np.ndarray],
                        channel_idx: int, rng=None) -> np.ndarray:
    """用对立类样本的同通道同频段替换目标频段。"""
    if rng is None:
        rng = np.random.default_rng()
    donor_idx = rng.integers(0, len(other_class_signals))
    donor_signal_1d = other_class_signals[donor_idx][channel_idx]
    donor_band = _bandpass_filter(donor_signal_1d, fs, f_lo, f_hi)
    original_band = _bandpass_filter(signal_1d, fs, f_lo, f_hi)
    residual = signal_1d - original_band
    return (residual + donor_band).astype(signal_1d.dtype)

 
def _gaussian_replace_band(signal_1d: np.ndarray, fs: float,
                           f_lo: float, f_hi: float, rng=None) -> np.ndarray:
    """用 1/f 频谱形状的噪声替换目标频段，保持 EEG 自然的功率衰减特性。"""
    if rng is None:
        rng = np.random.default_rng()
    band_component = _bandpass_filter(signal_1d, fs, f_lo, f_hi)
    band_std = np.std(band_component)
    if band_std < 1e-10:
        return signal_1d.copy()
    # 生成白噪声并在频域施加 1/f 包络
    n = len(signal_1d)
    white = rng.normal(0, 1, size=n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # 1/f 包络（避免 DC 除零）
    envelope = np.ones_like(freqs)
    envelope[1:] = 1.0 / np.sqrt(freqs[1:])
    fft_colored = fft_white * envelope
    colored = np.fft.irfft(fft_colored, n=n)
    # 带通滤波到目标频段，再缩放到原始频段能量
    colored_band = _bandpass_filter(colored.astype(signal_1d.dtype), fs, f_lo, f_hi)
    colored_std = np.std(colored_band)
    if colored_std > 1e-10:
        colored_band = colored_band * (band_std / colored_std)
    residual = signal_1d - band_component
    return (residual + colored_band).astype(signal_1d.dtype)


def _apply_band_mask(signal: np.ndarray, fs: float,
                     channel: int, f_lo: float, f_hi: float,
                     baseline_mode: str = 'phase_shuffle', rng=None,
                     other_class_signals: List[np.ndarray] = None) -> np.ndarray:
    """对 signal (n_ch, T) 的指定通道施加频段扰动，返回扰动后的 signal。"""
    out = signal.copy()
    if baseline_mode == 'zero':
        out[channel] = _bandstop_filter(signal[channel], fs, f_lo, f_hi)
    elif baseline_mode == 'phase_shuffle':
        out[channel] = _phase_shuffle_band(signal[channel], fs, f_lo, f_hi, rng=rng)
    elif baseline_mode == 'gaussian':
        out[channel] = _gaussian_replace_band(signal[channel], fs, f_lo, f_hi, rng=rng)
    elif baseline_mode == 'class_permute':
        if other_class_signals is None or len(other_class_signals) == 0:
            raise ValueError("class_permute requires non-empty other_class_signals")
        out[channel] = _class_permute_band(
            signal[channel], fs, f_lo, f_hi,
            other_class_signals, channel, rng=rng)
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")
    return out


def _apply_multi_band_mask(signal: np.ndarray, fs: float,
                           channel: int, band_mask: Dict[str, bool],
                           freq_bands: dict, baseline_mode: str = 'phase_shuffle',
                           rng=None,
                           other_class_signals: List[np.ndarray] = None) -> np.ndarray:
    """对指定通道同时施加多个频段扰动（band_mask[band]=True 表示去掉该频段）。"""
    out = signal.copy()
    for band_name, should_remove in band_mask.items():
        if should_remove:
            f_lo, f_hi = freq_bands[band_name]
            if baseline_mode == 'zero':
                out[channel] = _bandstop_filter(out[channel], fs, f_lo, f_hi)
            elif baseline_mode == 'phase_shuffle':
                out[channel] = _phase_shuffle_band(out[channel], fs, f_lo, f_hi, rng=rng)
            elif baseline_mode == 'gaussian':
                out[channel] = _gaussian_replace_band(out[channel], fs, f_lo, f_hi, rng=rng)
            elif baseline_mode == 'class_permute':
                if other_class_signals is None or len(other_class_signals) == 0:
                    raise ValueError("class_permute requires non-empty other_class_signals")
                out[channel] = _class_permute_band(
                    out[channel], fs, f_lo, f_hi,
                    other_class_signals, channel, rng=rng)
    return out


# ==================== 前向推理辅助 ====================

def _get_logit(adapter, signal: np.ndarray, target_class: int,
               device: str = 'cuda') -> float:
    """对 signal (n_ch, T) 做预处理 + 前向推理，返回 target_class 的 logit。"""
    tensor = adapter.prepare_input(signal)
    # 确保 tensor dtype 与模型一致
    model_dtype = next(adapter.model.parameters()).dtype
    if tensor.dtype != model_dtype:
        tensor = tensor.to(dtype=model_dtype)
    with torch.no_grad():
        output = adapter.forward(tensor)
    if output.numel() == 1:
        is_binary = getattr(adapter, 'is_binary', False)
        if is_binary and target_class == 0:
            return -output.item()
        return output.item()
    if output.dim() > 1:
        return output[0, target_class].item()
    return output[target_class].item()


# ==================== 单样本归因方法 ====================

def band_occlusion_single(
    raw_signal: np.ndarray,
    adapter,
    target_class: int,
    fs: float,
    freq_bands: dict = None,
    baseline_mode: str = 'phase_shuffle',
    rng=None,
    batch_size: int = 32,
    other_class_signals: List[np.ndarray] = None,
    n_permutations: int = 1,
) -> np.ndarray:
    """单样本 Occlusion 频段归因 → (n_channels, n_bands)
    所有 (channel, band) 扰动信号 batch 推理，大幅加速。
    class_permute 时每个 (ch, band) 重复 n_permutations 次取均值。"""
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    band_names = list(freq_bands.keys())
    n_ch = raw_signal.shape[0]
    n_bands = len(band_names)

    original_logit = _get_logit(adapter, raw_signal, target_class)
    if np.isnan(original_logit) or np.isinf(original_logit):
        print(f"    [Warning] original_logit is {original_logit}, signal shape={raw_signal.shape}")
        return np.zeros((raw_signal.shape[0], len(band_names)))

    is_binary = getattr(adapter, 'is_binary', False)

    n_reps = n_permutations if baseline_mode == 'class_permute' else 1
    result_accum = np.zeros((n_ch, n_bands))

    batch_signals = []
    batch_indices = []

    with torch.no_grad():
        for rep in range(n_reps):
            for bi, band_name in enumerate(band_names):
                f_lo, f_hi = freq_bands[band_name]
                for ch in range(n_ch):
                    perturbed = _apply_band_mask(raw_signal, fs, ch, f_lo, f_hi,
                                                 baseline_mode, rng=rng,
                                                 other_class_signals=other_class_signals)
                    batch_signals.append(perturbed)
                    batch_indices.append((ch, bi, rep))

                    if len(batch_signals) >= batch_size:
                        _occlusion_flush_permute(
                            batch_signals, batch_indices, adapter,
                            target_class, original_logit, is_binary,
                            result_accum, n_reps)
                        batch_signals = []
                        batch_indices = []

        if batch_signals:
            _occlusion_flush_permute(
                batch_signals, batch_indices, adapter,
                target_class, original_logit, is_binary,
                result_accum, n_reps)

    return result_accum


def _occlusion_flush(batch_signals, batch_indices, adapter, target_class,
                     original_logit, is_binary, result):
    tensor = adapter.prepare_input(np.stack(batch_signals, axis=0))
    model_dtype = next(adapter.model.parameters()).dtype
    if tensor.dtype != model_dtype:
        tensor = tensor.to(dtype=model_dtype)
    output = adapter.forward(tensor)
    if output.numel() == output.shape[0]:
        logits = output.squeeze().cpu().numpy()
        if is_binary and target_class == 0:
            logits = -logits
    elif output.dim() > 1:
        logits = output[:, target_class].cpu().numpy()
    else:
        logits = output.cpu().numpy()
    if logits.ndim == 0:
        logits = logits.reshape(1)
    for k, logit_val in enumerate(logits):
        ch, bi = batch_indices[k]
        result[ch, bi] = original_logit - float(logit_val)
    del tensor, output


def _occlusion_flush_permute(batch_signals, batch_indices, adapter, target_class,
                              original_logit, is_binary, result_accum, n_reps):
    """flush 支持多次置换平均：batch_indices 中每个元素为 (ch, bi, rep)。"""
    tensor = adapter.prepare_input(np.stack(batch_signals, axis=0))
    model_dtype = next(adapter.model.parameters()).dtype
    if tensor.dtype != model_dtype:
        tensor = tensor.to(dtype=model_dtype)
    output = adapter.forward(tensor)
    if output.numel() == output.shape[0]:
        logits = output.squeeze().cpu().numpy()
        if is_binary and target_class == 0:
            logits = -logits
    elif output.dim() > 1:
        logits = output[:, target_class].cpu().numpy()
    else:
        logits = output.cpu().numpy()
    if logits.ndim == 0:
        logits = logits.reshape(1)
    for k, logit_val in enumerate(logits):
        ch, bi, rep = batch_indices[k]
        result_accum[ch, bi] += (original_logit - float(logit_val)) / n_reps
    del tensor, output


def band_shap_single(
    raw_signal: np.ndarray,
    adapter,
    target_class: int,
    fs: float,
    freq_bands: dict = None,
    baseline_mode: str = 'phase_shuffle',
    rng=None,
    batch_size: int = 32,
    other_class_signals: List[np.ndarray] = None,
    n_permutations: int = 1,
) -> np.ndarray:
    """单样本 SHAP (Shapley value) 频段归因 → (n_channels, n_bands)
    每个通道独立计算 5 频段的 Shapley 值。所有组合 batch 推理。"""
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    band_names = list(freq_bands.keys())
    n_ch = raw_signal.shape[0]
    n_bands = len(band_names)
    result = np.zeros((n_ch, n_bands))
    is_binary = getattr(adapter, 'is_binary', False)

    for ch in range(n_ch):
        all_coalitions = []
        all_signals = []
        for size in range(n_bands + 1):
            for combo in combinations(range(n_bands), size):
                coalition = frozenset(combo)
                band_mask = {bn: (bi in coalition) for bi, bn in enumerate(band_names)}
                perturbed = _apply_multi_band_mask(
                    raw_signal, fs, ch, band_mask, freq_bands, baseline_mode,
                    rng=rng, other_class_signals=other_class_signals)
                all_coalitions.append(coalition)
                all_signals.append(perturbed)

        coalition_logits = {}
        with torch.no_grad():
            for start in range(0, len(all_signals), batch_size):
                batch_signals = np.stack(all_signals[start:start + batch_size], axis=0)
                tensor = adapter.prepare_input(batch_signals)
                model_dtype = next(adapter.model.parameters()).dtype
                if tensor.dtype != model_dtype:
                    tensor = tensor.to(dtype=model_dtype)
                output = adapter.forward(tensor)

                if output.numel() == output.shape[0]:
                    logits = output.squeeze().cpu().numpy()
                    if is_binary and target_class == 0:
                        logits = -logits
                elif output.dim() > 1:
                    logits = output[:, target_class].cpu().numpy()
                else:
                    logits = output.cpu().numpy()

                if logits.ndim == 0:
                    logits = logits.reshape(1)

                for k, logit_val in enumerate(logits):
                    coalition_logits[all_coalitions[start + k]] = float(logit_val)
                del tensor, output

        from math import factorial
        for bi, band_name in enumerate(band_names):
            shapley_val = 0.0
            others = [j for j in range(n_bands) if j != bi]
            for size in range(n_bands):
                for combo in combinations(others, size):
                    S = frozenset(combo)
                    S_with_i = S | {bi}
                    marginal = coalition_logits[S] - coalition_logits[S_with_i]
                    weight = factorial(size) * factorial(n_bands - size - 1) / factorial(n_bands)
                    shapley_val += weight * marginal
            result[ch, bi] = shapley_val

    return result


def band_ig_single(
    raw_signal: np.ndarray,
    adapter,
    target_class: int,
    fs: float,
    freq_bands: dict = None,
    baseline_mode: str = 'phase_shuffle',
    n_steps: int = 20,
    rng=None,
    other_class_signals: List[np.ndarray] = None,
    n_permutations: int = 1,
) -> np.ndarray:
    """单样本 IG 频段归因 → (n_channels, n_bands)
    对每个 (channel, band)，从 baseline(去掉该频段) 到原始信号做积分。
    所有插值步合并成一个 batch 一次性 forward+backward，大幅加速。"""
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    band_names = list(freq_bands.keys())
    n_ch = raw_signal.shape[0]
    n_bands = len(band_names)
    result = np.zeros((n_ch, n_bands))

    alphas = np.linspace(0, 1, n_steps + 1)

    for bi, band_name in enumerate(band_names):
        f_lo, f_hi = freq_bands[band_name]
        for ch in range(n_ch):
            baseline = _apply_band_mask(raw_signal, fs, ch, f_lo, f_hi,
                                         baseline_mode, rng=rng,
                                         other_class_signals=other_class_signals)
            diff = raw_signal - baseline

            interp_batch = np.stack(
                [baseline + a * diff for a in alphas], axis=0
            )
            tensor = adapter.prepare_input(interp_batch)
            model_dtype = next(adapter.model.parameters()).dtype
            tensor = tensor.clone().detach().to(dtype=model_dtype).requires_grad_(True)

            output = adapter.forward(tensor)
            if output.numel() == output.shape[0]:
                scores = output.squeeze()
                is_binary = getattr(adapter, 'is_binary', False)
                if is_binary and target_class == 0:
                    scores = -scores
                target_output = scores.sum()
            elif output.dim() > 1:
                target_output = output[:, target_class].sum()
            else:
                target_output = output.sum()
            target_output.backward()

            grad = tensor.grad.detach().cpu().numpy()
            if grad.ndim == 4:
                grad = grad.reshape(grad.shape[0], n_ch, -1)
            avg_grad = grad[:, ch, :].mean(axis=0)
            result[ch, bi] = (avg_grad * diff[ch]).sum()

            del tensor, output, target_output, grad, interp_batch, baseline, diff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def band_lime_single(
    raw_signal: np.ndarray,
    adapter,
    target_class: int,
    fs: float,
    freq_bands: dict = None,
    baseline_mode: str = 'phase_shuffle',
    n_samples: int = 100,
    rng=None,
    batch_size: int = 32,
    other_class_signals: List[np.ndarray] = None,
    n_permutations: int = 1,
) -> np.ndarray:
    """单样本 LIME 频段归因 → (n_channels, n_bands)
    每个通道独立：随机开关频段组合，拟合线性模型。batch 推理加速。"""
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    if rng is None:
        rng = np.random.default_rng()
    band_names = list(freq_bands.keys())
    n_ch = raw_signal.shape[0]
    n_bands = len(band_names)
    result = np.zeros((n_ch, n_bands))
    is_binary = getattr(adapter, 'is_binary', False)

    for ch in range(n_ch):
        masks = rng.integers(0, 2, size=(n_samples, n_bands))

        all_perturbed = []
        for si in range(n_samples):
            band_mask = {bn: (masks[si, bi] == 0) for bi, bn in enumerate(band_names)}
            perturbed = _apply_multi_band_mask(
                raw_signal, fs, ch, band_mask, freq_bands, baseline_mode,
                rng=rng, other_class_signals=other_class_signals)
            all_perturbed.append(perturbed)

        logits = np.zeros(n_samples)
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                batch_signals = np.stack(all_perturbed[start:start + batch_size], axis=0)
                tensor = torch.from_numpy(batch_signals).float().to(adapter.device)
                if tensor.dim() == 3:
                    tensor = adapter.to_patch_input(tensor)
                model_dtype = next(adapter.model.parameters()).dtype
                if tensor.dtype != model_dtype:
                    tensor = tensor.to(dtype=model_dtype)
                output = adapter.forward(tensor)

                if output.numel() == output.shape[0]:
                    batch_logits = output.squeeze().cpu().numpy()
                    if is_binary and target_class == 0:
                        batch_logits = -batch_logits
                elif output.dim() > 1:
                    batch_logits = output[:, target_class].cpu().numpy()
                else:
                    batch_logits = output.cpu().numpy()

                if batch_logits.ndim == 0:
                    batch_logits = batch_logits.reshape(1)

                logits[start:start + len(batch_logits)] = batch_logits
                del tensor, output

        del all_perturbed
        distances = np.sum(masks == 0, axis=1).astype(float)
        weights = np.exp(-distances / n_bands)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(masks, logits, sample_weight=weights)
        result[ch] = model.coef_

    return result


def band_gradient_shap_single(
    raw_signal: np.ndarray,
    adapter,
    target_class: int,
    fs: float,
    freq_bands: dict = None,
    baseline_mode: str = 'phase_shuffle',
    n_baselines: int = 10,
    n_steps: int = 10,
    rng=None,
    other_class_signals: List[np.ndarray] = None,
    n_permutations: int = 1,
) -> np.ndarray:
    """单样本 Gradient×SHAP 频段归因 → (n_channels, n_bands)

    对每个 (channel, band)，用多个随机基线跑 IG 然后取均值。
    所有 baselines 的插值步合并为一个大 batch 一次性 forward+backward，
    相比逐 baseline 串行循环，速度提升约 n_baselines 倍。
    """
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    if rng is None:
        rng = np.random.default_rng()
    band_names = list(freq_bands.keys())
    n_ch = raw_signal.shape[0]
    n_bands = len(band_names)
    result = np.zeros((n_ch, n_bands))
    is_binary = getattr(adapter, 'is_binary', False)

    n_alpha = n_steps + 1
    alphas = np.linspace(0, 1, n_alpha)

    for bi, band_name in enumerate(band_names):
        f_lo, f_hi = freq_bands[band_name]
        for ch in range(n_ch):
            # ---- 生成所有 baselines 的插值点，合并为一个大 batch ----
            all_interp = []          # n_baselines × n_alpha 个信号
            all_diffs_ch = []        # 每个 baseline 对应的 diff[ch]
            for _ in range(n_baselines):
                baseline = _apply_band_mask(
                    raw_signal, fs, ch, f_lo, f_hi, baseline_mode, rng=rng,
                    other_class_signals=other_class_signals)
                diff = raw_signal - baseline
                all_diffs_ch.append(diff[ch])
                for a in alphas:
                    all_interp.append(baseline + a * diff)

            # 大 batch forward + backward
            big_batch = np.stack(all_interp, axis=0)  # (n_baselines*n_alpha, C, T)
            tensor = torch.from_numpy(big_batch).float().to(adapter.device)
            if tensor.dim() == 3:
                tensor = adapter.to_patch_input(tensor)
            model_dtype = next(adapter.model.parameters()).dtype
            if tensor.dtype != model_dtype:
                tensor = tensor.to(dtype=model_dtype)
            tensor = tensor.clone().detach().requires_grad_(True)

            output = adapter.forward(tensor)
            total = n_baselines * n_alpha
            if output.numel() == total:
                scores = output.squeeze()
                if is_binary and target_class == 0:
                    scores = -scores
            elif output.dim() > 1:
                scores = output[:, target_class]
            else:
                scores = output
            loss = scores.sum()
            loss.backward()

            grad = tensor.grad.detach().cpu().numpy()  # (n_baselines*n_alpha, ...)
            del tensor, output, scores, loss, big_batch

            if grad.ndim > 3:
                grad = grad.reshape(total, n_ch, -1)

            # ---- 按 baseline 分组，计算每个 baseline 的 IG 值 ----
            ig_values = []
            for bi_idx in range(n_baselines):
                start = bi_idx * n_alpha
                end = start + n_alpha
                avg_grad = grad[start:end, ch, :].mean(axis=0)
                ig_values.append((avg_grad * all_diffs_ch[bi_idx]).sum())

            result[ch, bi] = np.mean(ig_values)
            del grad, all_interp, all_diffs_ch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ==================== 功率占比计算 ====================

def _compute_band_power_ratio(
    raw_signals: List[np.ndarray],
    fs: float,
    freq_bands: dict,
) -> np.ndarray:
    """计算各频段的平均功率占比 → (n_channels, n_bands)

    对所有样本算 Welch PSD，然后按频段积分，归一化为占比。
    用于给 LLM 提供参考：归因值需要结合 power_ratio 判断是否受 1/f 影响。
    """
    band_names = list(freq_bands.keys())
    n_ch = raw_signals[0].shape[0]
    n_bands = len(band_names)

    # 累加所有样本的频段功率
    total_power = np.zeros((n_ch, n_bands))
    for sig in raw_signals:
        for ch in range(n_ch):
            freqs, psd = welch(sig[ch], fs=fs, nperseg=min(256, len(sig[ch])))
            for bi, (bn, (f_lo, f_hi)) in enumerate(freq_bands.items()):
                mask = (freqs >= f_lo) & (freqs < f_hi)
                total_power[ch, bi] += psd[mask].sum()

    # 每个通道归一化为占比
    ch_totals = total_power.sum(axis=1, keepdims=True)
    ch_totals = np.maximum(ch_totals, 1e-10)  # 避免除零
    power_ratio = total_power / ch_totals  # (n_ch, n_bands)

    return power_ratio


# ==================== 群体聚合 ====================

BAND_METHOD_MAP = {
    'occlusion': band_occlusion_single,
    'shap': band_shap_single,
    'ig': band_ig_single,
    'lime': band_lime_single,
    'gradient_shap': band_gradient_shap_single,
}


def grand_band_attribution(
    raw_signals: List[np.ndarray],
    adapter,
    target_class: int,
    fs: float,
    freq_bands: dict = None,
    methods: List[str] = None,
    baseline_mode: str = 'auto',
    model_type: Optional[str] = None,
    channel_names: List[str] = None,
    grand_channel_importance: np.ndarray = None,
    top_k: int = 5,
    output_dir: str = './band_attribution',
    save: bool = True,
    show: bool = False,
    seed: int = 42,
    n_workers: int = 1,
    other_class_signals: List[np.ndarray] = None,
    n_permutations: int = 20,
) -> Dict:
    """
    群体频段归因入口。

    对每个样本跑指定方法的频段归因，然后跨样本聚合统计。
    n_workers>1 时使用多线程并行（滤波 CPU 部分释放 GIL）。

    other_class_signals: 对立类样本列表，class_permute 模式必需。
    n_permutations: class_permute 时每个 (ch, band) 的重复次数（默认 20）。
    """
    # Auto-resolve baseline_mode based on model architecture
    if baseline_mode == 'auto':
        baseline_mode = resolve_band_baseline('auto', model_type, other_class_signals)
        print(f"[Band Attribution] auto baseline → '{baseline_mode}' (model_type={model_type})")

    if baseline_mode == 'class_permute' and (other_class_signals is None or len(other_class_signals) == 0):
        print("[Band Attribution] class_permute requested but no other_class_signals, falling back to 'zero'")
        baseline_mode = 'zero'

    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    if methods is None:
        methods = ['occlusion', 'shap']
    os.makedirs(output_dir, exist_ok=True)

    band_names = list(freq_bands.keys())
    n_samples = len(raw_signals)
    n_ch = raw_signals[0].shape[0]
    n_bands = len(band_names)
    rng = np.random.default_rng(seed)

    print(f"\n[Band Attribution] methods={methods}, baseline={baseline_mode}, "
          f"n_samples={n_samples}, n_ch={n_ch}, n_bands={n_bands}, n_workers={n_workers}"
          + (f", n_permutations={n_permutations}" if baseline_mode == 'class_permute' else ""))

    all_results = {}

    for method in methods:
        if method not in BAND_METHOD_MAP:
            print(f"  [Warning] Unknown method: {method}, skipping.")
            continue

        func = BAND_METHOD_MAP[method]
        print(f"\n  Running {method.upper()}...")
        per_sample = np.zeros((n_samples, n_ch, n_bands))

        import time as _time
        t0 = _time.time()

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            sample_rngs = [np.random.default_rng(seed + si) for si in range(n_samples)]

            def _run_single(si):
                return si, func(
                    raw_signal=raw_signals[si], adapter=adapter,
                    target_class=target_class, fs=fs, freq_bands=freq_bands,
                    baseline_mode=baseline_mode, rng=sample_rngs[si],
                    other_class_signals=other_class_signals,
                    n_permutations=n_permutations,
                )

            completed = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_run_single, si): si for si in range(n_samples)}
                for future in as_completed(futures):
                    si, result = future.result()
                    per_sample[si] = result
                    completed += 1
                    elapsed = _time.time() - t0
                    s = per_sample[si]
                    print(f"    {completed}/{n_samples} done ({elapsed:.1f}s) | "
                          f"band_abs_mean={np.mean(np.abs(s), axis=0).tolist()}", flush=True)
        else:
            for si, sig in enumerate(raw_signals):
                per_sample[si] = func(
                    raw_signal=sig, adapter=adapter, target_class=target_class,
                    fs=fs, freq_bands=freq_bands, baseline_mode=baseline_mode,
                    rng=rng,
                    other_class_signals=other_class_signals,
                    n_permutations=n_permutations,
                )
                elapsed = _time.time() - t0
                s = per_sample[si]
                print(f"    {si+1}/{n_samples} done ({elapsed:.1f}s) | "
                      f"band_abs_mean={np.mean(np.abs(s), axis=0).tolist()}", flush=True)

        # 群体统计（不做逐样本归一化，直接取均值）
        grand_mean = np.mean(per_sample, axis=0)  # (n_ch, n_bands)
        grand_sem = np.std(per_sample, axis=0) / np.sqrt(n_samples)

        # 全局频段重要性：对通道取绝对值均值
        band_importance = np.mean(np.abs(grand_mean), axis=0)  # (n_bands,)

        # 频段视角的通道重要性：对频段取绝对值之和
        channel_importance = np.sum(np.abs(grand_mean), axis=1)  # (n_ch,)
        top_k_ch_idx = np.argsort(channel_importance)[::-1][:top_k]
        top_k_channels = []
        if channel_names is not None:
            top_k_channels = [(channel_names[i], float(channel_importance[i]))
                              for i in top_k_ch_idx]
        else:
            top_k_channels = [(f"ch{i}", float(channel_importance[i]))
                              for i in top_k_ch_idx]

        # 每个频段的 p 值（Wilcoxon 检验：该频段的归因是否显著非零）
        band_pvalues = {}
        for bi, bn in enumerate(band_names):
            vals = per_sample[:, :, bi].mean(axis=1)  # 每个样本对通道取均值
            if np.all(vals == 0):
                band_pvalues[bn] = 1.0
            else:
                try:
                    _, p = wilcoxon(vals)
                    band_pvalues[bn] = float(p)
                except ValueError:
                    band_pvalues[bn] = 1.0

        all_results[method] = {
            'method': method,
            'baseline_mode': baseline_mode,
            'per_sample': per_sample,
            'channel_band_importance': grand_mean,
            'channel_band_sem': grand_sem,
            'band_importance': band_importance,
            'band_pvalues': band_pvalues,
            'channel_importance': channel_importance,
            'top_k_channels': top_k_channels,
            'band_names': band_names,
            'channel_names': channel_names or [f"ch{i}" for i in range(n_ch)],
            'n_samples': n_samples,
        }

        print(f"  {method.upper()} done. Band ranking: "
              f"{[band_names[i] for i in np.argsort(band_importance)[::-1]]}")

    # 方法间一致性
    cross_consistency = {}
    method_keys = list(all_results.keys())

    if not method_keys:
        print("  [Warning] No valid band attribution methods ran. Skipping.")
        return None

    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            m1, m2 = method_keys[i], method_keys[j]
            flat1 = all_results[m1]['channel_band_importance'].flatten()
            flat2 = all_results[m2]['channel_band_importance'].flatten()
            r, p = spearmanr(flat1, flat2)
            cross_consistency[f"{m1}_vs_{m2}"] = {'spearman_r': float(r), 'p_value': float(p)}
            print(f"  Consistency {m1} vs {m2}: Spearman r={r:.3f}, p={p:.4f}")

    # 频段 vs 时间归因通道对比
    spatial_consistency = None
    if grand_channel_importance is not None and channel_names is not None:
        temporal_top_idx = np.argsort(grand_channel_importance)[::-1][:top_k]
        temporal_top_names = [channel_names[i] for i in temporal_top_idx]
        # 用第一个方法的结果
        first_method = method_keys[0]
        band_top_names = [name for name, _ in all_results[first_method]['top_k_channels']]
        overlap = list(set(temporal_top_names) & set(band_top_names))
        union = list(set(temporal_top_names) | set(band_top_names))
        jaccard = len(overlap) / len(union) if union else 0.0
        spatial_consistency = {
            'temporal_top_k': temporal_top_names,
            'band_top_k': band_top_names,
            'jaccard': jaccard,
            'overlap': overlap,
        }
        print(f"  Spatial consistency: Jaccard={jaccard:.2f}, overlap={overlap}")

    # 计算各频段功率占比（供 LLM 分析时参考，消除 1/f 误判）
    power_ratio = _compute_band_power_ratio(raw_signals, fs, freq_bands)
    # 全局功率占比：对所有通道取平均 → (n_bands,)
    global_power_ratio = power_ratio.mean(axis=0)
    print(f"  Band power ratio: {dict(zip(band_names, [f'{v:.3f}' for v in global_power_ratio]))}")

    # 可视化 + JSON
    if save or show:
        plot_band_importance_bar(all_results, freq_bands, output_dir, save, show)
        for method in method_keys:
            plot_channel_band_heatmap(all_results[method], freq_bands, output_dir, save, show)
            plot_band_topomap(all_results[method], freq_bands, output_dir, save, show)
        if spatial_consistency is not None:
            plot_spatial_consistency(spatial_consistency, output_dir, save, show)
        if len(method_keys) > 1:
            plot_method_consistency(cross_consistency, output_dir, save, show)

    if save:
        save_band_attribution_json(
            all_results=all_results,
            freq_bands=freq_bands,
            cross_consistency=cross_consistency,
            spatial_consistency=spatial_consistency,
            target_class=target_class,
            baseline_mode=baseline_mode,
            output_dir=output_dir,
            power_ratio=power_ratio,
            global_power_ratio=global_power_ratio,
        )

    return {
        'results': all_results,
        'cross_consistency': cross_consistency,
        'spatial_consistency': spatial_consistency,
    }


# ==================== 功率对比 ====================

def compute_band_power_context(
    raw_signals_target: List[np.ndarray],
    raw_signals_other: List[np.ndarray],
    fs: float,
    freq_bands: dict = None,
    channel_names: List[str] = None,
    top_channel_band_pairs: List[dict] = None,
) -> Dict:
    """
    计算目标类和其他类的绝对频段功率。
    返回 global_band_power + by_channel_band (仅 top pairs)。
    """
    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    band_names = list(freq_bands.keys())

    def _compute_band_powers(signals):
        """返回 (n_ch, n_bands) 的平均功率"""
        all_powers = []
        for sig in signals:
            n_ch = sig.shape[0]
            powers = np.zeros((n_ch, len(band_names)))
            for bi, (bn, (f_lo, f_hi)) in enumerate(freq_bands.items()):
                for ch in range(n_ch):
                    band_sig = _bandpass_filter(sig[ch], fs, f_lo, f_hi)
                    powers[ch, bi] = np.mean(band_sig ** 2)
            all_powers.append(powers)
        return np.mean(np.stack(all_powers), axis=0)

    target_powers = _compute_band_powers(raw_signals_target)
    other_powers = _compute_band_powers(raw_signals_other)

    global_band_power = {}
    for bi, bn in enumerate(band_names):
        global_band_power[bn] = {
            'target': float(np.mean(target_powers[:, bi])),
            'other': float(np.mean(other_powers[:, bi])),
        }

    by_channel_band = []
    if top_channel_band_pairs and channel_names:
        ch_name_to_idx = {n: i for i, n in enumerate(channel_names)}
        band_name_to_idx = {n: i for i, n in enumerate(band_names)}
        for pair in top_channel_band_pairs:
            ch_idx = ch_name_to_idx.get(pair['channel'])
            bi = band_name_to_idx.get(pair['band'])
            if ch_idx is not None and bi is not None:
                by_channel_band.append({
                    'channel': pair['channel'],
                    'band': pair['band'],
                    'target_class_power': float(target_powers[ch_idx, bi]),
                    'other_class_power': float(other_powers[ch_idx, bi]),
                })

    return {
        'global_band_power': global_band_power,
        'by_channel_band': by_channel_band,
    }


def plot_psd_comparison(
    raw_signals_by_class: Dict[int, List[np.ndarray]],
    fs: float,
    freq_bands: dict = None,
    channel_names: List[str] = None,
    top_channels: List[str] = None,
    class_labels: Dict[int, str] = None,
    output_dir: str = '.',
    save: bool = True,
    show: bool = False,
):
    """画各类别 PSD 对比图。每个通道一个子图，每张图 2×2=4 通道，自动分页。"""
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from scipy.stats import mannwhitneyu

    if freq_bands is None:
        freq_bands = DEFAULT_BANDS
    band_names = list(freq_bands.keys())

    # 防御：如果任意类别的信号列表为空，跳过绘图
    if not raw_signals_by_class or any(len(v) == 0 for v in raw_signals_by_class.values()):
        print("  [plot_psd_comparison] Skipped: one or more classes have zero samples.")
        return []

    # 获取所有通道列表
    n_ch = raw_signals_by_class[list(raw_signals_by_class.keys())[0]][0].shape[0]
    ch_names = channel_names or [f"ch{i}" for i in range(n_ch)]

    if class_labels is None:
        class_labels = {c: f"class {c}" for c in raw_signals_by_class}

    ch_name_to_idx = {ch: i for i, ch in enumerate(ch_names)}

    # 每张图 2×2 = 4 个通道
    channels_per_fig = 4
    n_figs = (n_ch + channels_per_fig - 1) // channels_per_fig

    n_classes = len(raw_signals_by_class)
    class_list = sorted(raw_signals_by_class.keys())
    colors = {class_list[0]: '#1f77b4', class_list[1]: '#d62728'} if n_classes == 2 else \
             {cls: plt.cm.tab10(i) for i, cls in enumerate(class_list)}

    # 频段颜色（淡色背景）
    band_colors_map = {
        'delta': '#e8f4f8',
        'theta': '#d5e8f7',
        'alpha': '#c8e6c9',
        'beta': '#fff9c4',
        'gamma': '#ffccbc',
    }

    for fig_idx in range(n_figs):
        start_ch = fig_idx * channels_per_fig
        end_ch = min(start_ch + channels_per_fig, n_ch)
        n_ch_this_fig = end_ch - start_ch

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for sub_idx in range(channels_per_fig):
            ax = axes[sub_idx]

            if start_ch + sub_idx >= n_ch:
                ax.set_visible(False)
                continue

            ch_name = ch_names[start_ch + sub_idx]
            ch_idx = ch_name_to_idx[ch_name]

            # 收集各类 PSD 用于统计检验
            psd_by_class = {}
            freqs = None

            for cls, signals in sorted(raw_signals_by_class.items()):
                all_psd = []
                for sig in signals:
                    if sig.ndim < 2 or ch_idx >= sig.shape[0]:
                        continue
                    f, psd = welch(sig[ch_idx], fs=fs, nperseg=min(len(sig[ch_idx]), int(fs * 2)))
                    all_psd.append(psd)
                    if freqs is None:
                        freqs = f

                if not all_psd:
                    continue

                all_psd_arr = np.array(all_psd)
                psd_by_class[cls] = all_psd_arr

                median_psd = np.median(all_psd_arr, axis=0)
                q25 = np.percentile(all_psd_arr, 25, axis=0)
                q75 = np.percentile(all_psd_arr, 75, axis=0)

                # 画曲线（线性 Y 轴）
                ax.plot(freqs, median_psd, color=colors[cls],
                       label=f"{class_labels[cls]} (n={len(signals)})", linewidth=1.8)
                ax.fill_between(freqs, q25, q75, color=colors[cls], alpha=0.15)

            # 统计检验：逐频点 Mann-Whitney U
            if len(psd_by_class) == 2 and freqs is not None:
                cls0, cls1 = class_list
                psd0 = psd_by_class.get(cls0)
                psd1 = psd_by_class.get(cls1)

                if psd0 is not None and psd1 is not None and len(psd0) > 0 and len(psd1) > 0:
                    # 只测 0.5-45Hz 范围
                    freq_mask = (freqs >= 0.5) & (freqs <= 45)
                    freqs_test = freqs[freq_mask]

                    p_values = []
                    for fi in np.where(freq_mask)[0]:
                        try:
                            _, p = mannwhitneyu(psd0[:, fi], psd1[:, fi], alternative='two-sided')
                            p_values.append(p)
                        except:
                            p_values.append(1.0)

                    p_values = np.array(p_values)
                    sig_mask = p_values < 0.05

                    # 画显著性阴影
                    ylim = ax.get_ylim()
                    for fi, is_sig in enumerate(sig_mask):
                        if is_sig:
                            ax.axvspan(freqs_test[fi] - 0.25, freqs_test[fi] + 0.25,
                                      color='gray', alpha=0.2, zorder=0)

            # 频段背景色带
            for bn, (f_lo, f_hi) in freq_bands.items():
                color = band_colors_map.get(bn, '#f0f0f0')
                ax.axvspan(f_lo, f_hi, alpha=0.12, color=color, zorder=0)

            # 顶部频段标注
            ylim = ax.get_ylim()
            y_pos = ylim[1] * 0.95
            for bn, (f_lo, f_hi) in freq_bands.items():
                ax.text((f_lo + f_hi) / 2, y_pos, bn,
                       ha='center', va='top', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.7))

            ax.set_title(ch_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=9)
            ax.set_ylabel('PSD (μV²/Hz)', fontsize=9)
            ax.set_xlim(0.5, 45)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # 只在第一个子图显示图例
            if sub_idx == 0 and ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=8)

        fig.suptitle(f'PSD Comparison by Class (Channels {start_ch+1}-{end_ch})',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        if save:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f'psd_comparison_page{fig_idx+1}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  [Saved] {path}")
        if show:
            plt.show()
        plt.close(fig)

    return None  # 返回 None（多张图）


# ==================== 可视化 ====================

def plot_band_importance_bar(all_results: Dict, freq_bands: dict,
                             output_dir: str, save: bool, show: bool):
    import matplotlib.pyplot as plt

    band_names = list(freq_bands.keys())
    methods = list(all_results.keys())
    n_methods = len(methods)
    n_bands = len(band_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.7 / n_methods
    x = np.arange(n_bands)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    for mi, method in enumerate(methods):
        r = all_results[method]
        means = r['band_importance']
        sems = np.std(r['per_sample'].mean(axis=1), axis=0) / np.sqrt(r['n_samples'])
        offset = (mi - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=sems, capsize=3,
                      label=method.upper(), color=colors[mi % len(colors)], alpha=0.85)
        # 标注显著性
        for bi, bn in enumerate(band_names):
            p = r['band_pvalues'].get(bn, 1.0)
            if p < 0.001:
                ax.text(x[bi] + offset, means[bi] + sems[bi] + 0.01, '***',
                        ha='center', va='bottom', fontsize=8)
            elif p < 0.01:
                ax.text(x[bi] + offset, means[bi] + sems[bi] + 0.01, '**',
                        ha='center', va='bottom', fontsize=8)
            elif p < 0.05:
                ax.text(x[bi] + offset, means[bi] + sems[bi] + 0.01, '*',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{bn}\n({freq_bands[bn][0]}-{freq_bands[bn][1]}Hz)"
                        for bn in band_names])
    ax.set_ylabel('Band Importance (|attribution|)')
    ax.set_title('Frequency Band Attribution')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'band_importance_bar.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [Saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_channel_band_heatmap(result: Dict, freq_bands: dict,
                              output_dir: str, save: bool, show: bool):
    import matplotlib.pyplot as plt

    band_names = result['band_names']
    ch_names = result['channel_names']
    matrix = result['channel_band_importance']  # (n_ch, n_bands)
    method = result['method']

    # 按区域排序通道
    regions = ['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital', 'Other']
    sorted_indices = []
    sorted_names = []
    region_boundaries = []
    for region in regions:
        region_chs = [(i, ch) for i, ch in enumerate(ch_names) if _ch_region(ch) == region]
        if region_chs:
            region_boundaries.append((len(sorted_indices), region))
            for idx, name in region_chs:
                sorted_indices.append(idx)
                sorted_names.append(name)

    sorted_matrix = matrix[sorted_indices]

    fig, ax = plt.subplots(figsize=(6, max(4, len(ch_names) * 0.3)))
    vmax = np.abs(sorted_matrix).max()
    im = ax.imshow(sorted_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels([f"{bn}\n({freq_bands[bn][0]}-{freq_bands[bn][1]}Hz)"
                        for bn in band_names], fontsize=8)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=7)

    # 区域分隔线
    for pos, region in region_boundaries:
        if pos > 0:
            ax.axhline(pos - 0.5, color='gray', linewidth=0.5, linestyle='--')
        ax.text(-0.8, pos + 0.3, region, fontsize=7, fontweight='bold',
                ha='right', va='top', transform=ax.get_yaxis_transform())

    plt.colorbar(im, ax=ax, label='Attribution (signed)', shrink=0.8)
    ax.set_title(f'Channel × Band Attribution ({method.upper()})')
    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, f'channel_band_heatmap_{method}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [Saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_band_topomap(result: Dict, freq_bands: dict,
                      output_dir: str, save: bool, show: bool,
                      top_k: int = 5, metadata: dict = None):
    """每个频段一张 topomap，标注 top-K 电极名，标题含方法/baseline/样本数。"""
    try:
        import mne
    except ImportError:
        print("  [Warning] mne not installed, skipping band topomap.")
        return

    band_names = result['band_names']
    ch_names = result['channel_names']
    matrix = result['channel_band_importance']
    method = result['method']
    baseline_mode = result.get('baseline_mode', '?')
    n_samples = result.get('n_samples', '?')
    n_bands = len(band_names)

    try:
        from explainability.visualizer import resolve_channel_positions
        montage = mne.channels.make_standard_montage('standard_1020')
        valid_names, valid_indices, pos_2d, has_bipolar = resolve_channel_positions(
            ch_names, montage)
    except Exception as e:
        print(f"  [Warning] Failed to resolve channel positions: {e}")
        return

    if len(valid_names) < 3:
        print(f"  [Warning] Only {len(valid_names)} channels matched montage, skipping topomap.")
        return

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    fig, axes = plt.subplots(1, n_bands, figsize=(n_bands * 2.8, 3.8))
    if n_bands == 1:
        axes = [axes]

    valid_matrix = matrix[valid_indices]
    vmax = np.abs(valid_matrix).max() + 1e-8

    for bi, (bn, ax) in enumerate(zip(band_names, axes)):
        data = valid_matrix[:, bi]
        norm_data = data / vmax

        try:
            mne.viz.plot_topomap(norm_data, pos_2d, axes=ax, show=False,
                                 vlim=(-1, 1), cmap='RdBu_r',
                                 contours=6)
        except (AttributeError, TypeError):
            try:
                mne.viz.plot_topomap(norm_data, pos_2d, axes=ax, show=False,
                                     vlim=(-1, 1), cmap='RdBu_r',
                                     contours=0)
            except ValueError:
                mne.viz.plot_topomap(norm_data, pos_2d, axes=ax, show=False,
                                     vlim=(-1, 1), cmap='RdBu_r',
                                     contours=0, sensors=False)

        if top_k > 0 and len(data) > 0:
            sorted_idx = np.argsort(np.abs(data))[::-1][:min(top_k, len(data))]
            for si in sorted_idx:
                x, y = pos_2d[si]
                ax.text(x, y, valid_names[si], ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))

        subtitle = f"{bn}\n({freq_bands[bn][0]}-{freq_bands[bn][1]}Hz)"
        if has_bipolar:
            subtitle += "\n[bipolar midpoint]"
        ax.set_title(subtitle, fontsize=13, fontweight='bold')

    title = f"Band Topomap — {method.upper()} | baseline={baseline_mode} | n={n_samples}"
    if metadata:
        meta_parts = []
        if metadata.get('model') is not None:
            meta_parts.append(f"Model: {metadata['model']}")
        if metadata.get('method') is not None:
            meta_parts.append(f"Method: {metadata['method']}")
        if metadata.get('label') is not None:
            meta_parts.append(f"Label: {metadata['label']}")
        if metadata.get('prediction') is not None:
            meta_parts.append(f"Pred: {metadata['prediction']}")
        if metadata.get('confidence') is not None:
            meta_parts.append(f"Prob: {metadata['confidence']:.4f}")
        if metadata.get('label') is not None and metadata.get('prediction') is not None:
            correct = metadata['label'] == metadata['prediction']
            meta_parts.append(f"{'Correct' if correct else 'Incorrect'}")
        if meta_parts:
            title += '\n' + ' | '.join(meta_parts)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, f'band_topomap_{method}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  [Saved] {path} (+pdf)")
    if show:
        plt.show()
    plt.close(fig)


def plot_spatial_consistency(spatial_consistency: Dict,
                             output_dir: str, save: bool, show: bool):
    import matplotlib.pyplot as plt
    try:
        from matplotlib_venn import venn2
        has_venn = True
    except ImportError:
        has_venn = False

    temporal = set(spatial_consistency['temporal_top_k'])
    band = set(spatial_consistency['band_top_k'])

    fig, ax = plt.subplots(figsize=(6, 4))
    if has_venn:
        venn2([temporal, band], set_labels=('Temporal Top-K', 'Band Top-K'), ax=ax)
    else:
        ax.text(0.5, 0.5, f"Temporal: {spatial_consistency['temporal_top_k']}\n"
                f"Band: {spatial_consistency['band_top_k']}\n"
                f"Overlap: {spatial_consistency['overlap']}\n"
                f"Jaccard: {spatial_consistency['jaccard']:.2f}",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title(f"Spatial Consistency (Jaccard={spatial_consistency['jaccard']:.2f})")
    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'spatial_consistency.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [Saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_method_consistency(cross_consistency: Dict,
                            output_dir: str, save: bool, show: bool):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    pairs = list(cross_consistency.keys())
    rs = [cross_consistency[p]['spearman_r'] for p in pairs]
    ps = [cross_consistency[p]['p_value'] for p in pairs]

    colors = ['#55A868' if p < 0.05 else '#C44E52' for p in ps]
    bars = ax.barh(range(len(pairs)), rs, color=colors, alpha=0.85)

    for i, (r, p) in enumerate(zip(rs, ps)):
        ax.text(r + 0.02, i, f"r={r:.3f}, p={p:.4f}", va='center', fontsize=9)

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p.replace('_vs_', ' vs ').upper() for p in pairs])
    ax.set_xlabel('Spearman Correlation')
    ax.set_title('Cross-Method Consistency')
    ax.set_xlim(-0.2, 1.1)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'method_consistency.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [Saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


# ==================== JSON 输出 ====================

def save_band_attribution_json(
    all_results: Dict,
    freq_bands: dict,
    cross_consistency: Dict,
    spatial_consistency: Optional[Dict],
    target_class: int,
    baseline_mode: str,
    output_dir: str,
    task: str = '',
    model: str = '',
    sample_type: str = 'TP',
    n_samples: int = 0,
    power_context: Optional[Dict] = None,
    power_ratio: Optional[np.ndarray] = None,
    global_power_ratio: Optional[np.ndarray] = None,
):
    band_names = list(freq_bands.keys())

    methods_json = {}
    for method, r in all_results.items():
        ch_names = r['channel_names']
        grand_mean = r['channel_band_importance']
        n_ch = grand_mean.shape[0]

        # band_importance
        band_imp = {}
        for bi, bn in enumerate(band_names):
            entry = {
                'mean': float(r['band_importance'][bi]),
                'sem': float(r['channel_band_sem'][:, bi].mean()),
                'p_value': float(r['band_pvalues'].get(bn, 1.0)),
            }
            if global_power_ratio is not None:
                entry['power_ratio'] = float(global_power_ratio[bi])
                # 归因密度：归因值 / 功率占比，反映单位能量的贡献
                pr = max(float(global_power_ratio[bi]), 0.01)
                entry['attribution_density'] = float(r['band_importance'][bi] / pr)
            band_imp[bn] = entry

        # band ranking (by absolute importance)
        band_ranking = [band_names[i] for i in np.argsort(r['band_importance'])[::-1]]

        # top channel-band pairs (by absolute value)
        flat_idx = np.argsort(np.abs(grand_mean).flatten())[::-1]
        top_pairs = []
        for rank, idx in enumerate(flat_idx[:10]):
            ch_i = idx // len(band_names)
            bi = idx % len(band_names)
            pair = {
                'channel': ch_names[ch_i],
                'band': band_names[bi],
                'importance': float(grand_mean[ch_i, bi]),
                'rank': rank + 1,
            }
            if power_ratio is not None:
                pair['power_ratio'] = float(power_ratio[ch_i, bi])
            top_pairs.append(pair)

        # channel importance from band perspective
        ch_imp = {ch_names[i]: float(r['channel_importance'][i]) for i in range(n_ch)}

        # laterality by band
        laterality = {}
        for bi, bn in enumerate(band_names):
            left_vals = [grand_mean[i, bi] for i, ch in enumerate(ch_names)
                         if _ch_laterality(ch) == 'Left']
            right_vals = [grand_mean[i, bi] for i, ch in enumerate(ch_names)
                          if _ch_laterality(ch) == 'Right']
            if left_vals and right_vals:
                l_mean = np.mean(np.abs(left_vals))
                r_mean = np.mean(np.abs(right_vals))
                denom = l_mean + r_mean
                asym = (r_mean - l_mean) / denom if denom > 0 else 0.0
                laterality[bn] = {
                    'asymmetry': float(asym),
                    'direction': 'right_dominant' if asym > 0 else 'left_dominant',
                }

        # region by band
        region_by_band = {}
        for bi, bn in enumerate(band_names):
            region_vals = {}
            for region in REGION_MAP:
                vals = [np.abs(grand_mean[i, bi]) for i, ch in enumerate(ch_names)
                        if _ch_region(ch) == region]
                if vals:
                    region_vals[region.lower()] = float(np.mean(vals))
            if region_vals:
                region_by_band[bn] = region_vals

        methods_json[method] = {
            'band_importance': band_imp,
            'band_ranking': band_ranking,
            'top_channel_band_pairs': top_pairs,
            'channel_importance_from_band': ch_imp,
            'laterality_by_band': laterality,
            'region_by_band': region_by_band,
        }

    out = {
        'analysis_type': 'band_attribution',
        'task': task,
        'model': model,
        'target_class': target_class,
        'sample_type': sample_type,
        'n_samples': n_samples or (list(all_results.values())[0]['n_samples'] if all_results else 0),
        'baseline_mode': baseline_mode,
        'freq_bands': {bn: list(v) for bn, v in freq_bands.items()},
        'methods': methods_json,
        'cross_method_consistency': cross_consistency,
    }

    # 全局功率占比 + 1/f 说明
    if global_power_ratio is not None:
        out['band_power_ratio'] = {
            bn: float(global_power_ratio[bi])
            for bi, bn in enumerate(band_names)
        }
        out['band_power_ratio_note'] = (
            "EEG signals have a natural 1/f power spectrum: lower frequency bands "
            "carry more power. band_power_ratio shows each band's share of total power. "
            "When interpreting band_importance, compare it with power_ratio: "
            "if importance/power_ratio (attribution_density) is similar across bands, "
            "the model has no specific band preference beyond energy. "
            "If a band's attribution_density is significantly higher, "
            "the model specifically relies on that band's structure for classification."
        )

    if spatial_consistency is not None:
        out['spatial_consistency_with_temporal'] = spatial_consistency

    if power_context is not None:
        out['power_context'] = power_context

    path = os.path.join(output_dir, 'band_attribution_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  [Saved] {path}")
