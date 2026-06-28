"""
逆向频谱归因分析
Spectral Attribution Analysis for EEG Explainability

基于 run_explainability 得到的 result['combined'] (n_channels, n_patches) 权重，
对 Top-K 通道做 STFT 时频分析，识别驱动分类结果的神经节律。

输出:
  1. 归因时频热力图（每个 Top-K 通道一张，原始功率谱 + 归因加权叠加）
  2. 频段贡献占比直方图（X轴: Top-K 通道, Y轴: 贡献占比, 颜色: 频段）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft
from scipy.interpolate import interp1d
from typing import List, Optional, Dict
import os


# ==================== 频段定义 ====================

FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 45),
}

# ==================== 脑区 / 侧化映射（与 analysis.py 保持一致）====================

REGION_MAP = {
    "Frontal":   {"Fp1","Fp2","F3","F4","F7","F8","Fz","AF3","AF4","AF7","AF8","AFz"},
    "Temporal":  {"T3","T4","T5","T6","T7","T8","TP7","TP8","FT7","FT8"},
    "Central":   {"C3","C4","Cz","FC1","FC2","FC3","FC4","FC5","FC6","FCz"},
    "Parietal":  {"P3","P4","Pz","P7","P8","CP1","CP2","CP3","CP4","CP5","CP6","CPz"},
    "Occipital": {"O1","O2","Oz","PO3","PO4","PO7","PO8"},
}

LATERALITY_LEFT  = {"F7","F3","T3","T7","C3","P3","P7","T5","O1","Fp1",
                    "FC1","FC3","FC5","FT7","CP1","CP3","CP5","TP7","PO3","PO7","AF3","AF7"}
LATERALITY_RIGHT = {"F8","F4","T4","T8","C4","P4","P8","T6","O2","Fp2",
                    "FC2","FC4","FC6","FT8","CP2","CP4","CP6","TP8","PO4","PO8","AF4","AF8"}


def _ch_region(ch: str) -> str:
    u = ch.upper().split('-')[0]
    for region, names in REGION_MAP.items():
        if u in {n.upper() for n in names}:
            return region
    return "Other"


def _ch_laterality(ch: str) -> str:
    u = ch.upper().split('-')[0]
    if u in {n.upper() for n in LATERALITY_LEFT}:  return "Left"
    if u in {n.upper() for n in LATERALITY_RIGHT}: return "Right"
    return "Midline"

BAND_COLORS = {
    'Delta': '#4C72B0',
    'Theta': '#DD8452',
    'Alpha': '#55A868',
    'Beta':  '#C44E52',
    'Gamma': '#8172B2',
}


# ==================== 核心分析函数 ====================

def _plot_grand_avg_topomap(
    grand_channel_importance: np.ndarray,
    channel_names: List[str],
    top_pos: np.ndarray,
    top_neg: np.ndarray,
    n_samples: int,
    output_dir: str,
    save: bool,
    show: bool,
    subtitle: str = '',
    title_prefix: str = '',
):
    """地形图：群体平均通道重要度，图下方标注正/负 Top-K 通道名"""
    top_pos_names = [channel_names[i] for i in top_pos]
    top_neg_names = [channel_names[i] for i in top_neg] if len(top_neg) > 0 else []

    try:
        import mne
        from explainability.visualizer import EEGVisualizer, resolve_channel_positions

        montage = mne.channels.make_standard_montage('standard_1020')
        valid_names, valid_indices, pos_2d, has_bipolar = resolve_channel_positions(
            channel_names, montage)

        if len(valid_names) == 0:
            print("  [Warning] No standard channels found, falling back to bar chart")
            _plot_grand_avg_importance_bar(
                grand_channel_importance, channel_names, top_pos_names,
                n_samples, output_dir, save, show
            )
            return

        valid_importance = np.array([grand_channel_importance[i] for i in valid_indices])
        abs_max = np.abs(valid_importance).max() + 1e-8
        importance_norm = valid_importance / abs_max

        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
            fig, ax = plt.subplots(figsize=(8, 9))
            im, _ = EEGVisualizer._plot_topomap_safe(
                importance_norm, pos_2d, ax, cmap='RdBu_r',
                vlim=(-1, 1), names=valid_names
            )
            # MNE 默认电极名 ~6pt；用 findobj 抓所有 Text 对象（含子 artist）
            import matplotlib.text as _mtext
            name_set = set(valid_names)
            for t in fig.findobj(_mtext.Text):
                if t.get_text() in name_set:
                    t.set_fontsize(13)
                    t.set_fontweight('bold')
            ax.set_title(
                (f'{title_prefix}\n' if title_prefix else f'Grand Average Channel Importance  (n={n_samples} samples)\n')
                + (f'{subtitle}\n' if subtitle and not title_prefix else '')
                + 'Red=positive attribution, Blue=negative attribution',
                fontsize=16, fontweight='bold', pad=6
            )
            cb = plt.colorbar(im, ax=ax, shrink=0.5, orientation='horizontal',
                              pad=0.03, label='Importance')
            cb.ax.tick_params(labelsize=12)
            cb.set_label('Importance', fontsize=14)

            pos_label = f'Top-{len(top_pos)} positive: ' + ', '.join(top_pos_names)
            fig.text(0.5, 0.05, pos_label, ha='center', va='bottom',
                     fontsize=14, color='#C44E52',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#C44E52', alpha=0.8))
            if top_neg_names:
                neg_label = f'Top-{len(top_neg)} negative: ' + ', '.join(top_neg_names)
                fig.text(0.5, 0.01, neg_label, ha='center', va='bottom',
                         fontsize=14, color='#4878CF',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='#4878CF', alpha=0.8))

            plt.subplots_adjust(top=0.96, bottom=0.13, left=0.0, right=1.0)
            if save:
                path = os.path.join(output_dir, 'grand_avg_topomap.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
                print(f"[Saved] {path} (+pdf)")
            if show:
                plt.show()
            plt.close()

        except Exception as e:
            print(f"  [Warning] MNE topomap failed: {e}, falling back to bar chart")
            _plot_grand_avg_importance_bar(
                grand_channel_importance, channel_names, top_pos_names,
                n_samples, output_dir, save, show
            )

    except ImportError:
        print("  [Warning] MNE not available, using bar chart instead")
        _plot_grand_avg_importance_bar(
            grand_channel_importance, channel_names, top_pos_names,
            n_samples, output_dir, save, show
        )

    # 保存 topomap 数值为 JSON（与画图逻辑无关，始终执行）
    if save:
        import json
        # 解析 subtitle 中的结构化信息（格式: "TP | class 1 | method: GRADCAM | n=40 conf≥0.7"）
        meta: Dict = {"analysis_type": "population", "n_samples": n_samples}
        for part in subtitle.split("|"):
            part = part.strip()
            if part in ("TP", "FP"):
                meta["sample_type"] = part
            elif part.startswith("class "):
                try:
                    meta["target_class"] = int(part.split()[-1])
                except ValueError:
                    pass
            elif part.startswith("method:"):
                meta["method"] = part.split(":", 1)[-1].strip()
            elif part.startswith("n="):
                tokens = part.split()
                for tok in tokens:
                    if tok.startswith("n="):
                        try:
                            meta["n_samples"] = int(tok[2:])
                        except ValueError:
                            pass
                    elif "conf" in tok:
                        meta["conf_threshold"] = tok.replace("conf≥", "").replace("conf>=", "").strip()

        topomap_data = {
            "meta": meta,
            "top_positive_channels": top_pos_names,
            "top_negative_channels": top_neg_names,
            "channel_importance": {
                ch: round(float(v), 6)
                for ch, v in zip(channel_names, grand_channel_importance)
            },
        }
        json_path = os.path.join(output_dir, 'topomap_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(topomap_data, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {json_path}")


def _plot_grand_avg_importance_bar(
    grand_channel_importance: np.ndarray,
    channel_names: List[str],
    top_channel_names: List[str],
    n_samples: int,
    output_dir: str,
    save: bool,
    show: bool,
):
    """地形图失败时的降级方案：通道重要度条形图，Top-K 通道高亮"""
    indices = np.argsort(grand_channel_importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [
        '#C44E52' if channel_names[i] in top_channel_names else '#4C72B0'
        for i in indices
    ]
    ax.barh(
        [channel_names[i] for i in indices],
        grand_channel_importance[indices],
        color=bar_colors
    )
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance')
    ax.set_title(f'Grand Average Channel Importance (n={n_samples} samples)')
    top_label = f'Top-{len(top_channel_names)} channels: ' + ', '.join(top_channel_names)
    fig.text(0.5, 0.01, top_label, ha='center', va='bottom', fontsize=10,
             color='#C44E52')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save:
        path = os.path.join(output_dir, 'grand_avg_channel_importance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


def compute_patch_band_correlation(
    raw_signal: np.ndarray,        # (n_channels, signal_length)
    combined: np.ndarray,          # (n_channels, n_patches) Patch 归因权重
    channel_names: List[str],
    patch_size: int,
    fs: float,
    spatial_importance: Optional[np.ndarray] = None,  # (n_channels,) 用于选 Top-K 通道
    top_k: int = 5,
    output_dir: str = './patch_band_correlation',
    save: bool = True,
    show: bool = False,
) -> Dict:
    """
    单样本 Patch 归因与频段能量波动的 Spearman 相关性分析。

    对每个通道：
      1. 用小波变换计算每个 Patch 内各频段的平均能量 → band_energy(n_patches, n_bands)
      2. 计算 Patch 归因分数与各频段能量的 Spearman 相关系数 → (n_bands,)

    输出：
      - corr_matrix : (n_channels, n_bands) 相关系数矩阵
      - pval_matrix : (n_channels, n_bands) p 值矩阵
      - 热力图: 通道 × 频段相关系数
      - 柱状图: 每个频段在所有通道上的相关系数分布
      - Top-K 通道折线图: 每个 Top-K 通道一张子图，横轴频段，纵轴相关系数

    Args:
        raw_signal        : (n_channels, signal_length)
        combined          : (n_channels, n_patches) Patch 归因权重
        channel_names     : 通道名列表
        patch_size        : 每个 Patch 的采样点数
        fs                : 采样率 (Hz)
        spatial_importance: (n_channels,) 通道重要性，用于选 Top-K；None 时用 combined 行均值
        top_k             : Top-K 通道数，用于单通道频段相关图
        output_dir        : 输出目录
        save/show         : 控制图片

    Returns:
        dict with 'corr_matrix' (n_channels, n_bands), 'pval_matrix', 'band_names', 'channel_names'
    """
    import pywt
    from scipy.stats import spearmanr

    os.makedirs(output_dir, exist_ok=True)

    # 如果是 patch 化格式先还原
    if raw_signal.ndim == 3:
        raw_signal = raw_signal.reshape(raw_signal.shape[0], -1)

    n_channels, signal_length = raw_signal.shape
    n_patches = combined.shape[1]
    band_names = list(FREQ_BANDS.keys())
    n_bands = len(band_names)

    # 构建小波 scales → 覆盖 0.5~45 Hz
    scales = np.geomspace(
        pywt.central_frequency('cmor1-1.5') * fs / 45,
        pywt.central_frequency('cmor1-1.5') * fs / 0.5,
        num=128
    )
    freqs = pywt.scale2frequency('cmor1-1.5', scales) * fs  # (128,) 降序

    corr_matrix = np.zeros((n_channels, n_bands))
    pval_matrix = np.ones((n_channels, n_bands))

    for ch_idx in range(n_channels):
        signal = raw_signal[ch_idx].astype(float)
        patch_weights = combined[ch_idx]  # (n_patches,)

        # 整段小波变换
        coeffs, _ = pywt.cwt(signal, scales, 'cmor1-1.5', sampling_period=1.0 / fs)
        power = np.abs(coeffs) ** 2  # (n_scales, signal_length)

        n_t = power.shape[1]

        # 每个 Patch 内各频段平均能量 → band_energy (n_patches, n_bands)
        band_energy = np.zeros((n_patches, n_bands))
        for p in range(n_patches):
            t_start = int(p * n_t / n_patches)
            t_end = int((p + 1) * n_t / n_patches)
            if t_end <= t_start:
                t_end = t_start + 1
            patch_power = power[:, t_start:t_end].mean(axis=1)  # (n_scales,)
            for b_idx, (band_name, (flo, fhi)) in enumerate(FREQ_BANDS.items()):
                mask = (freqs >= flo) & (freqs <= fhi)
                band_energy[p, b_idx] = patch_power[mask].mean() if mask.any() else 0.0

        # Spearman 相关：patch 归因 vs 各频段能量
        for b_idx in range(n_bands):
            energy_vec = band_energy[:, b_idx]
            # 两列相同则相关无意义，跳过
            if energy_vec.std() < 1e-10 or patch_weights.std() < 1e-10:
                continue
            rho, pval = spearmanr(patch_weights, energy_vec)
            corr_matrix[ch_idx, b_idx] = rho
            pval_matrix[ch_idx, b_idx] = pval

    # ---- 可视化 ----
    _plot_patch_band_corr_heatmap(
        corr_matrix=corr_matrix,
        pval_matrix=pval_matrix,
        channel_names=channel_names,
        band_names=band_names,
        output_dir=output_dir,
        save=save,
        show=show,
    )
    _plot_patch_band_corr_bar(
        corr_matrix=corr_matrix,
        pval_matrix=pval_matrix,
        channel_names=channel_names,
        band_names=band_names,
        output_dir=output_dir,
        save=save,
        show=show,
    )

    # Top-K 通道逐通道频段相关图
    if spatial_importance is None:
        spatial_importance = np.abs(combined).mean(axis=1)
    top_indices = np.argsort(spatial_importance)[::-1][:top_k]
    _plot_topk_channel_band_corr(
        corr_matrix=corr_matrix,
        pval_matrix=pval_matrix,
        top_indices=top_indices,
        channel_names=channel_names,
        band_names=band_names,
        output_dir=output_dir,
        save=save,
        show=show,
    )

    print(f"[Patch-Band Correlation] corr_matrix shape: {corr_matrix.shape}")
    for b_idx, band_name in enumerate(band_names):
        top_ch_idx = np.argsort(np.abs(corr_matrix[:, b_idx]))[::-1][:3]
        top_info = ", ".join(
            f"{channel_names[i]}={corr_matrix[i, b_idx]:.3f}"
            for i in top_ch_idx
        )
        print(f"  {band_name}: top-3 channels → {top_info}")

    return {
        'corr_matrix': corr_matrix,
        'pval_matrix': pval_matrix,
        'band_names': band_names,
        'channel_names': channel_names,
    }


def _plot_patch_band_corr_heatmap(
    corr_matrix: np.ndarray,   # (n_channels, n_bands)
    pval_matrix: np.ndarray,
    channel_names: List[str],
    band_names: List[str],
    output_dir: str,
    save: bool,
    show: bool,
    pval_threshold: float = 0.05,
):
    """
    热力图：X 轴频段，Y 轴通道，颜色为 Spearman 相关系数。
    显著相关（p < pval_threshold）格子加星号。
    """
    n_ch, n_bands = corr_matrix.shape
    fig_h = max(6, n_ch * 0.35)
    fig, ax = plt.subplots(figsize=(n_bands * 1.4, fig_h))

    im = ax.imshow(
        corr_matrix,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        interpolation='nearest',
        origin='upper',
    )
    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(band_names, fontsize=11)
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels(channel_names, fontsize=max(5, 9 - n_ch // 20))
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(
        'Patch Attribution vs Band Energy\nSpearman Correlation (★ p<0.05)',
        fontsize=13, fontweight='bold'
    )

    # 显著格子加星号
    for ch_i in range(n_ch):
        for b_i in range(n_bands):
            if pval_matrix[ch_i, b_i] < pval_threshold:
                ax.text(b_i, ch_i, '★', ha='center', va='center',
                        fontsize=7, color='black', alpha=0.7)

    plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.6)
    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, 'patch_band_corr_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


def _plot_patch_band_corr_bar(
    corr_matrix: np.ndarray,   # (n_channels, n_bands)
    pval_matrix: np.ndarray,
    channel_names: List[str],
    band_names: List[str],
    output_dir: str,
    save: bool,
    show: bool,
    top_k: int = 10,
    pval_threshold: float = 0.05,
):
    """
    柱状图：每个频段一个子图，展示 Top-K（按 |ρ| 排序）通道的相关系数。
    显著通道（p < 0.05）柱子用深色，其余浅色。
    """
    n_bands = len(band_names)
    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 5), squeeze=False)
    fig.suptitle(
        'Patch Attribution vs Band Energy (Spearman ρ)\nTop channels per band',
        fontsize=13, y=1.02
    )

    for b_idx, band_name in enumerate(band_names):
        ax = axes[0, b_idx]
        rho_vec = corr_matrix[:, b_idx]
        pval_vec = pval_matrix[:, b_idx]

        # 按 |ρ| 降序取 Top-K
        order = np.argsort(np.abs(rho_vec))[::-1][:top_k]
        rho_top = rho_vec[order]
        pval_top = pval_vec[order]
        ch_top = [channel_names[i] for i in order]

        colors = [
            BAND_COLORS[band_name] if p < pval_threshold
            else '#CCCCCC'
            for p in pval_top
        ]
        bars = ax.barh(
            np.arange(len(ch_top)),
            rho_top,
            color=colors,
            edgecolor='white',
            linewidth=0.5,
        )
        ax.set_yticks(np.arange(len(ch_top)))
        ax.set_yticklabels(ch_top, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Spearman ρ', fontsize=9)
        ax.set_title(band_name, fontsize=11, color=BAND_COLORS[band_name],
                     fontweight='bold')

        # 数值标注
        for bar, rho, pval in zip(bars, rho_top, pval_top):
            sign_str = '★' if pval < pval_threshold else ''
            x_pos = bar.get_width()
            ha = 'left' if x_pos >= 0 else 'right'
            ax.text(
                x_pos + (0.03 if x_pos >= 0 else -0.03),
                bar.get_y() + bar.get_height() / 2,
                f'{rho:.2f}{sign_str}',
                ha=ha, va='center', fontsize=7,
            )

    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, 'patch_band_corr_bar.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


def _plot_topk_channel_band_corr(
    corr_matrix: np.ndarray,   # (n_channels, n_bands)
    pval_matrix: np.ndarray,
    top_indices: np.ndarray,   # Top-K 通道的索引（按 spatial_importance 排序）
    channel_names: List[str],
    band_names: List[str],
    output_dir: str,
    save: bool,
    show: bool,
    pval_threshold: float = 0.05,
):
    """
    Top-K 通道逐通道频段相关图。
    每个 Top-K 通道一个子图，横轴为频段，纵轴为 Spearman ρ，
    柱子颜色对应频段，显著（p<0.05）柱子深色，否则浅色。
    """
    top_k = len(top_indices)
    n_bands = len(band_names)
    x = np.arange(n_bands)

    fig, axes = plt.subplots(1, top_k, figsize=(3.5 * top_k, 4.5), squeeze=False)
    fig.suptitle(
        'Top-K Channel: Patch Attribution vs Band Energy (Spearman ρ)\n★ p<0.05',
        fontsize=12, y=1.02
    )

    for col, ch_idx in enumerate(top_indices):
        ax = axes[0, col]
        ch_name = channel_names[ch_idx]
        rho_vec = corr_matrix[ch_idx]   # (n_bands,)
        pval_vec = pval_matrix[ch_idx]  # (n_bands,)

        colors = [
            BAND_COLORS[band] if pval_vec[b] < pval_threshold else '#CCCCCC'
            for b, band in enumerate(band_names)
        ]
        bars = ax.bar(x, rho_vec, color=colors, edgecolor='white', linewidth=0.8, width=0.6)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=9)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Spearman ρ', fontsize=9)
        ax.set_title(f'{ch_name}\n(rank #{col + 1})', fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # 数值 + 星号标注
        for bar, rho, pval in zip(bars, rho_vec, pval_vec):
            sign = '★' if pval < pval_threshold else ''
            y_pos = rho + 0.04 if rho >= 0 else rho - 0.04
            va = 'bottom' if rho >= 0 else 'top'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f'{rho:.2f}{sign}',
                ha='center', va=va, fontsize=7.5,
            )

    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, 'patch_band_corr_topk_channels.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


# ==================== 群体级 Patch 归因-频段能量相关性分析 ====================

def grand_average_patch_band_correlation(
    raw_signals: List[np.ndarray],   # 列表，每个元素 (n_channels, signal_length)
    combineds: List[np.ndarray],     # 列表，每个元素 (n_channels, n_patches)
    channel_names: List[str],
    patch_size: int,
    fs: float,
    top_k: int = 5,
    n_freqs: int = 40,
    f_min: float = 0.5,
    f_max: float = 45.0,
    output_dir: str = './patch_band_correlation',
    save: bool = True,
    show: bool = False,
    confs: Optional[List[float]] = None,
    grand_channel_importance: Optional[np.ndarray] = None,  # 外部传入，与topomap/忠实度一致
    cwt_cache: Optional[Dict] = None,  # 外部传入的CWT缓存 {(sample_idx, ch_idx): (power, freqs)}
) -> Dict:
    """
    群体级 Patch 归因与频段瞬时功率的 Spearman 相关性分析。

    只对正贡献 top_k 通道和负贡献 top_k 通道计算 CWT（节省计算）。
    跨样本等权平均 + 单样本 t 检验（H0: ρ=0）评估显著性。

    输出：
      - pos_topk_channels.png : 正贡献 top_k 通道频段相关图（带 t 检验显著性）
      - neg_topk_channels.png : 负贡献 top_k 通道频段相关图
      - all_channels/page_XX.png : 全通道分页图（按重要度降序）

    Args:
        grand_channel_importance : 外部传入的通道重要度 (n_channels,)，有正有负。
                                   正值通道=支持预测，负值通道=抑制预测。
                                   None 时内部用 |combined| 均值估算（不推荐）。
        cwt_cache : 外部传入的CWT缓存字典，键为 (sample_idx, ch_idx)，值为 (power, freqs)。
                   传入时跳过CWT计算直接复用；同时本函数也会填充新计算的结果到此dict中。
    """
    import pywt
    from scipy.stats import spearmanr, ttest_1samp

    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(raw_signals)
    n_channels = len(channel_names)
    band_names = list(FREQ_BANDS.keys())
    n_bands = len(band_names)

    print(f"[Grand Patch-Band Correlation] n_samples={n_samples}, n_channels={n_channels}")

    # ---- Step 1: 确定正/负贡献 top_k 通道 ----
    if grand_channel_importance is not None:
        gci = grand_channel_importance
    else:
        gci = np.mean(np.stack([c.sum(axis=-1) for c in combineds], axis=0), axis=0)

    pos_mask = gci > 0
    neg_mask = gci < 0

    if pos_mask.any():
        pos_sorted = np.where(pos_mask)[0][np.argsort(gci[pos_mask])[::-1]]
    else:
        pos_sorted = np.array([], dtype=int)

    if neg_mask.any():
        neg_sorted = np.where(neg_mask)[0][np.argsort(gci[neg_mask])]  # 最负的在前
    else:
        neg_sorted = np.array([], dtype=int)

    pos_top_indices = pos_sorted[:top_k]
    neg_top_indices = neg_sorted[:top_k]
    target_indices = np.unique(np.concatenate([pos_top_indices, neg_top_indices])).astype(int)

    print(f"  Positive top-{len(pos_top_indices)}: {[channel_names[i] for i in pos_top_indices]}")
    print(f"  Negative top-{len(neg_top_indices)}: {[channel_names[i] for i in neg_top_indices]}")

    # 构建小波 scales（与 plot_wavelet_attribution_heatmap 统一，可复用缓存）
    w_morlet = 6.0
    f_axis = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)  # 低→高
    scales = w_morlet * fs / (2 * np.pi * f_axis)  # 低频=大尺度
    freqs = f_axis  # 直接用 f_axis，与 wavelet_heatmap 完全一致

    # ---- Step 2: 只对 target_indices 通道计算 corr，收集每个样本的 ρ ----
    # corr_collection[ch_idx][b_idx] = list of ρ across samples（所有patch）
    # pos_corr_collection[ch_idx][b_idx] = list of ρ（只用 combined > 0 的patch）
    # neg_corr_collection[ch_idx][b_idx] = list of ρ（只用 combined < 0 的patch）
    corr_collection     = {ch_idx: [[] for _ in range(n_bands)] for ch_idx in target_indices}
    pos_corr_collection = {ch_idx: [[] for _ in range(n_bands)] for ch_idx in target_indices}
    neg_corr_collection = {ch_idx: [[] for _ in range(n_bands)] for ch_idx in target_indices}

    for i, (raw_signal, combined) in enumerate(zip(raw_signals, combineds)):
        if raw_signal.ndim == 3:
            raw_signal = raw_signal.reshape(raw_signal.shape[0], -1)
        n_patches = combined.shape[1]

        for ch_idx in target_indices:
            signal = raw_signal[ch_idx].astype(float)
            patch_weights = combined[ch_idx]  # (n_patches,)

            # 使用缓存或计算CWT
            cache_key = (i, ch_idx)
            if cwt_cache is not None and cache_key in cwt_cache:
                power = cwt_cache[cache_key]
            else:
                coeffs, _ = pywt.cwt(signal, scales, 'cmor1-1.5', sampling_period=1.0 / fs)
                power = np.abs(coeffs) ** 2  # (n_freqs, signal_length)
                if cwt_cache is not None:
                    cwt_cache[cache_key] = power

            n_t = power.shape[1]

            band_energy = np.zeros((n_patches, n_bands))
            for p in range(n_patches):
                t_start = int(p * n_t / n_patches)
                t_end = int((p + 1) * n_t / n_patches)
                if t_end <= t_start:
                    t_end = t_start + 1
                patch_power = power[:, t_start:t_end].mean(axis=1)
                for b_idx, (_, (flo, fhi)) in enumerate(FREQ_BANDS.items()):
                    mask = (freqs >= flo) & (freqs <= fhi)
                    band_energy[p, b_idx] = patch_power[mask].mean() if mask.any() else 0.0

            # 正/负patch掩码
            pos_mask_p = patch_weights > 0
            neg_mask_p = patch_weights < 0

            for b_idx in range(n_bands):
                energy_vec = band_energy[:, b_idx]

                # 全patch ρ
                if energy_vec.std() < 1e-10 or patch_weights.std() < 1e-10:
                    corr_collection[ch_idx][b_idx].append(0.0)
                else:
                    rho, _ = spearmanr(patch_weights, energy_vec)
                    corr_collection[ch_idx][b_idx].append(float(rho))

                # 正patch ρ
                if pos_mask_p.sum() >= 3:
                    pw_pos = patch_weights[pos_mask_p]
                    ev_pos = energy_vec[pos_mask_p]
                    if pw_pos.std() > 1e-10 and ev_pos.std() > 1e-10:
                        rho_pos, _ = spearmanr(pw_pos, ev_pos)
                        pos_corr_collection[ch_idx][b_idx].append(float(rho_pos))

                # 负patch ρ（用归因绝对值，使"更负"对应"更大"）
                if neg_mask_p.sum() >= 3:
                    pw_neg = np.abs(patch_weights[neg_mask_p])
                    ev_neg = energy_vec[neg_mask_p]
                    if pw_neg.std() > 1e-10 and ev_neg.std() > 1e-10:
                        rho_neg, _ = spearmanr(pw_neg, ev_neg)
                        neg_corr_collection[ch_idx][b_idx].append(float(rho_neg))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_samples} samples done")

    # ---- Step 3: 跨样本均值 + t 检验 ----
    # grand_corr: (n_channels, n_bands)，未分析通道保持 0
    grand_corr = np.zeros((n_channels, n_bands))
    tstat_mat  = np.zeros((n_channels, n_bands))
    pval_mat   = np.ones((n_channels, n_bands))

    for ch_idx in target_indices:
        for b_idx in range(n_bands):
            rhos = np.array(corr_collection[ch_idx][b_idx])
            grand_corr[ch_idx, b_idx] = rhos.mean()
            if len(rhos) >= 3 and rhos.std() > 1e-10:
                tstat, pval = ttest_1samp(rhos, popmean=0.0)
                tstat_mat[ch_idx, b_idx] = tstat
                pval_mat[ch_idx, b_idx]  = pval

    print(f"  Grand corr computed. Pos top-3: "
          f"{[channel_names[i] for i in pos_top_indices[:3]]}")

    # ---- 诊断：打印每个样本的 ρ，用于判断跨样本方差来源 ----
    print(f"\n[Diagnosis] Per-sample Spearman ρ for pos/neg top channels:")
    for ch_idx in list(pos_top_indices[:3]) + list(neg_top_indices[:3]):
        ch_name = channel_names[ch_idx]
        sign = 'POS' if ch_idx in pos_top_indices else 'NEG'
        rho_by_band = []
        for b_idx, band_name in enumerate(band_names):
            rhos = np.array(corr_collection[ch_idx][b_idx])
            rho_str = ' '.join(f'{r:+.2f}' for r in rhos)
            rho_by_band.append(f"{band_name}=[{rho_str}] mean={rhos.mean():+.2f}")
        print(f"  [{sign}] {ch_name}: " + " | ".join(rho_by_band))

    # ---- Step 3b: 正/负patch分开的 grand corr + t检验 ----
    grand_pos_corr = np.zeros((n_channels, n_bands))
    grand_neg_corr = np.zeros((n_channels, n_bands))
    pval_pos_mat   = np.ones((n_channels, n_bands))
    pval_neg_mat   = np.ones((n_channels, n_bands))

    for ch_idx in target_indices:
        for b_idx in range(n_bands):
            pos_rhos = np.array(pos_corr_collection[ch_idx][b_idx])
            if len(pos_rhos) >= 3:
                grand_pos_corr[ch_idx, b_idx] = pos_rhos.mean()
                if pos_rhos.std() > 1e-10:
                    _, pval = ttest_1samp(pos_rhos, popmean=0.0)
                    pval_pos_mat[ch_idx, b_idx] = pval

            neg_rhos = np.array(neg_corr_collection[ch_idx][b_idx])
            if len(neg_rhos) >= 3:
                grand_neg_corr[ch_idx, b_idx] = neg_rhos.mean()
                if neg_rhos.std() > 1e-10:
                    _, pval = ttest_1samp(neg_rhos, popmean=0.0)
                    pval_neg_mat[ch_idx, b_idx] = pval

    # ---- Step 4: 正贡献 top_k 通道图 ----
    if len(pos_top_indices) > 0:
        _plot_grand_topk_channel_band_corr(
            grand_corr=grand_corr,
            pval_mat=pval_mat,
            top_indices=pos_top_indices,
            channel_names=channel_names,
            band_names=band_names,
            n_samples=n_samples,
            output_dir=output_dir,
            save=save,
            show=show,
            filename='pos_topk_channels.png',
            title_suffix='Positive Attribution',
        )

    # ---- Step 5: 负贡献 top_k 通道图 ----
    if len(neg_top_indices) > 0:
        _plot_grand_topk_channel_band_corr(
            grand_corr=grand_corr,
            pval_mat=pval_mat,
            top_indices=neg_top_indices,
            channel_names=channel_names,
            band_names=band_names,
            n_samples=n_samples,
            output_dir=output_dir,
            save=save,
            show=show,
            filename='neg_topk_channels.png',
            title_suffix='Negative Attribution',
        )

    # ---- Step 5b: 正/负patch分开的 ρ 图（每个通道上下两行）----
    all_target_sorted = np.array(
        [i for i in np.argsort(gci)[::-1] if i in set(target_indices.tolist())]
    )
    if len(all_target_sorted) > 0:
        _plot_signed_patch_band_corr(
            grand_pos_corr=grand_pos_corr,
            grand_neg_corr=grand_neg_corr,
            pval_pos_mat=pval_pos_mat,
            pval_neg_mat=pval_neg_mat,
            top_indices=all_target_sorted,
            channel_names=channel_names,
            band_names=band_names,
            n_samples=n_samples,
            output_dir=output_dir,
            save=save,
            show=show,
        )

    # ---- Step 6: 全通道分页图（按 gci 降序） ----
    all_sorted = np.argsort(gci)[::-1]
    # 只画有计算结果的通道（target_indices）
    all_sorted_filtered = np.array([i for i in all_sorted if i in set(target_indices.tolist())])
    if len(all_sorted_filtered) > 0:
        all_channels_dir = os.path.join(output_dir, 'all_channels')
        os.makedirs(all_channels_dir, exist_ok=True)
        _plot_all_channels_band_corr_paged(
            corr_matrix=grand_corr,
            pval_mat=pval_mat,
            sorted_indices=all_sorted_filtered,
            channel_names=channel_names,
            band_names=band_names,
            n_samples=n_samples,
            output_dir=all_channels_dir,
            save=save,
            show=show,
            channels_per_page=5,
        )

    return {
        'grand_corr': grand_corr,
        'pval_mat': pval_mat,
        'grand_spatial_importance': gci,
        'band_names': band_names,
        'channel_names': channel_names,
        'pos_top_indices': pos_top_indices,
        'neg_top_indices': neg_top_indices,
        'cwt_cache': cwt_cache,  # 返回（可能已填充的）缓存，供后续复用
        'scales': scales,
        'f_axis': f_axis,
        'target_indices': target_indices,
    }


def _plot_signed_patch_band_corr(
    grand_pos_corr: np.ndarray,   # (n_channels, n_bands) 正patch的ρ
    grand_neg_corr: np.ndarray,   # (n_channels, n_bands) 负patch的ρ（归因绝对值vs频段能量）
    pval_pos_mat: np.ndarray,
    pval_neg_mat: np.ndarray,
    top_indices: np.ndarray,      # 要画的通道索引（按重要度降序）
    channel_names: List[str],
    band_names: List[str],
    n_samples: int,
    output_dir: str,
    save: bool,
    show: bool,
    pval_threshold: float = 0.05,
):
    """
    正/负patch分开的频段相关图。
    每个通道一个子图，子图内上下两行：
      上行：正patch（combined > 0）内归因值 vs 频段能量的 Spearman ρ
      下行：负patch（combined < 0）内归因绝对值 vs 频段能量的 Spearman ρ
    ★ 标注跨样本 t 检验显著（p < pval_threshold）。
    """
    n_ch = len(top_indices)
    n_bands = len(band_names)
    x = np.arange(n_bands)

    fig, axes = plt.subplots(2, n_ch, figsize=(3.5 * n_ch, 7), squeeze=False)
    fig.suptitle(
        f'Signed Patch Band Correlation (Spearman ρ)  ★ p<{pval_threshold}\n'
        f'Top: positive patches  |  Bottom: negative patches  |  n={n_samples} samples',
        fontsize=12, y=1.02
    )

    for col, ch_idx in enumerate(top_indices):
        ch_name = channel_names[ch_idx]

        for row, (corr_row, pval_row, row_label) in enumerate([
            (grand_pos_corr[ch_idx], pval_pos_mat[ch_idx], 'Positive patches\n(combined > 0)'),
            (grand_neg_corr[ch_idx], pval_neg_mat[ch_idx], 'Negative patches\n(|combined| < 0)'),
        ]):
            ax = axes[row, col]
            colors = [BAND_COLORS[band] for band in band_names]
            bars = ax.bar(x, corr_row, color=colors, edgecolor='white',
                          linewidth=0.8, width=0.6, alpha=0.85)

            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, fontsize=9)
            ax.set_ylim(-1, 1)
            ax.set_ylabel('Spearman ρ', fontsize=9)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            if row == 0:
                ax.set_title(f'{ch_name}\n(rank #{col + 1})', fontsize=10, fontweight='bold')
            ax.set_xlabel(row_label, fontsize=8)

            for bar, rho, pval in zip(bars, corr_row, pval_row):
                sign = '★' if pval < pval_threshold else ''
                y_pos = rho + 0.04 if rho >= 0 else rho - 0.04
                va = 'bottom' if rho >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{rho:.2f}{sign}', ha='center', va=va, fontsize=7.5)

    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, 'signed_patch_band_corr.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


def _plot_all_channels_band_corr_paged(
    corr_matrix: np.ndarray,      # (n_channels, n_bands)
    pval_mat: np.ndarray,          # (n_channels, n_bands)
    sorted_indices: np.ndarray,   # 按重要度降序的通道索引
    channel_names: List[str],
    band_names: List[str],
    n_samples: int,
    output_dir: str,
    save: bool,
    show: bool,
    channels_per_page: int = 5,
    pval_threshold: float = 0.05,
):
    """全通道分页柱状图，每页 channels_per_page 个通道，显著频段加星号。"""
    n_channels = len(sorted_indices)
    n_bands = len(band_names)
    x = np.arange(n_bands)
    n_pages = (n_channels + channels_per_page - 1) // channels_per_page

    for page in range(n_pages):
        start = page * channels_per_page
        end = min(start + channels_per_page, n_channels)
        page_indices = sorted_indices[start:end]
        n_cols = len(page_indices)

        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.5), squeeze=False)
        fig.suptitle(
            f'Grand Average: Patch Attribution vs Band Energy (Spearman ρ)  ★ p<{pval_threshold}\n'
            f'n={n_samples} samples  |  Channels ranked by importance  '
            f'(page {page + 1}/{n_pages})',
            fontsize=11, y=1.02
        )

        for col, ch_idx in enumerate(page_indices):
            ax = axes[0, col]
            ch_name = channel_names[ch_idx]
            rho_vec = corr_matrix[ch_idx]
            pval_vec = pval_mat[ch_idx]
            rank = start + col + 1

            colors = [BAND_COLORS[band] for band in band_names]
            bars = ax.bar(x, rho_vec, color=colors, edgecolor='white',
                          linewidth=0.8, width=0.6, alpha=0.85)

            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, fontsize=9)
            ax.set_ylim(-1, 1)
            ax.set_ylabel('Spearman ρ', fontsize=9)
            ax.set_title(f'{ch_name}\n(rank #{rank})', fontsize=10, fontweight='bold')
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            for bar, rho, pval in zip(bars, rho_vec, pval_vec):
                sign = '★' if pval < pval_threshold else ''
                y_pos = rho + 0.04 if rho >= 0 else rho - 0.04
                va = 'bottom' if rho >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{rho:.2f}{sign}', ha='center', va=va, fontsize=7.5)

        plt.tight_layout()
        if save:
            path = os.path.join(output_dir, f'page_{page + 1:02d}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"[Saved] {path}")
        if show:
            plt.show()
        plt.close()


def _plot_grand_topk_channel_band_corr(
    grand_corr: np.ndarray,      # (n_channels, n_bands)
    pval_mat: np.ndarray,         # (n_channels, n_bands)
    top_indices: np.ndarray,
    channel_names: List[str],
    band_names: List[str],
    n_samples: int,
    output_dir: str,
    save: bool,
    show: bool,
    filename: str = 'topk_channels.png',
    title_suffix: str = '',
    pval_threshold: float = 0.05,
):
    """Top-K 通道群体平均频段相关柱状图，显著频段加星号。"""
    top_k = len(top_indices)
    n_bands = len(band_names)
    x = np.arange(n_bands)

    fig, axes = plt.subplots(1, top_k, figsize=(3.5 * top_k, 4.5), squeeze=False)
    suffix = f' — {title_suffix}' if title_suffix else ''
    fig.suptitle(
        f'Grand Average{suffix}: Patch Attribution vs Band Energy (Spearman ρ)  ★ p<{pval_threshold}\n'
        f'n={n_samples} samples',
        fontsize=12, y=1.02
    )

    for col, ch_idx in enumerate(top_indices):
        ax = axes[0, col]
        ch_name = channel_names[ch_idx]
        rho_vec = grand_corr[ch_idx]
        pval_vec = pval_mat[ch_idx]

        colors = [BAND_COLORS[band] for band in band_names]
        bars = ax.bar(x, rho_vec, color=colors, edgecolor='white',
                      linewidth=0.8, width=0.6, alpha=0.85)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=9)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Spearman ρ', fontsize=9)
        ax.set_title(f'{ch_name}\n(rank #{col + 1})', fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar, rho, pval in zip(bars, rho_vec, pval_vec):
            sign = '★' if pval < pval_threshold else ''
            y_pos = rho + 0.04 if rho >= 0 else rho - 0.04
            va = 'bottom' if rho >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{rho:.2f}{sign}', ha='center', va=va, fontsize=7.5)

    plt.tight_layout()
    if save:
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    if show:
        plt.show()
    plt.close()


# ==================== 小波归因时频热力图（逐通道，连续时间轴）====================

