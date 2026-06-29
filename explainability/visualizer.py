"""
EEG可视化工具类
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any
import os

 
# ── 通道坐标解析工具（topomap 共用）──────────────────────────────

# 电极别名映射（旧标准 / 大小写变体 → montage 标准名）
_ALIAS_MAP = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    'FPZ': 'Fpz', 'FP1': 'Fp1', 'FP2': 'Fp2',
    'AFZ': 'AFz',
    'FCZ': 'FCz',
    'CZ':  'Cz',
    'CPZ': 'CPz',
    'PZ':  'Pz',
    'POZ': 'POz',
    'OZ':  'Oz',
    'IZ':  'Iz',
    'FT9': 'FT9', 'FT10': 'FT10',
    'TP9': 'TP9', 'TP10': 'TP10',
    'PO9': 'PO9', 'PO10': 'PO10',
}

_REF_ELECTRODES = {'A1', 'A2', 'M1', 'M2', 'MASTL', 'MASTR'}


def _resolve_single_electrode(name_upper, standard_ch_upper, upper_to_montage):
    """将单个电极名（大写）解析为 montage 标准名，找不到返回 None"""
    if name_upper in standard_ch_upper:
        return upper_to_montage[name_upper]
    if name_upper in _ALIAS_MAP:
        candidate = _ALIAS_MAP[name_upper].upper()
        if candidate in standard_ch_upper:
            return upper_to_montage[candidate]
    return None


def resolve_channel_positions(ch_names, montage=None):
    """解析通道名列表为 topomap 可用的坐标。

    对单极通道：直接匹配 montage。
    对双极导联（含 '-'）：两端都找到 → 取中点；只找到一端 → 用该端；
        主电极是参考电极（A1/A2/M1/M2）→ 跳过。

    Returns:
        valid_names:    list[str]  — 原始通道名（用于标注）
        valid_indices:  list[int]  — 在 ch_names 中的下标
        positions:      ndarray (N, 2) — 2D 头皮坐标 (x, y)
        has_bipolar:    bool — 是否存在双极导联（调用方可据此决定是否标名）
    """
    import mne
    if montage is None:
        montage = mne.channels.make_standard_montage('standard_1020')

    standard_ch_upper = [ch.upper() for ch in montage.ch_names]
    upper_to_montage = {ch.upper(): ch for ch in montage.ch_names}

    # montage 3D 坐标 dict（标准名 → xyz）
    pos_3d = montage.get_positions()['ch_pos']

    valid_names = []
    valid_indices = []
    positions_3d = []
    has_bipolar = False
    seen_pos_keys = set()

    for i, ch_name in enumerate(ch_names):
        ch_upper = ch_name.upper()
        pos = None

        if '-' not in ch_upper:
            # 单极
            mapped = _resolve_single_electrode(ch_upper, standard_ch_upper, upper_to_montage)
            if mapped and mapped in pos_3d:
                key = mapped
                if key not in seen_pos_keys:
                    pos = pos_3d[mapped]
                    seen_pos_keys.add(key)
        else:
            # 双极
            parts = ch_upper.split('-')
            if len(parts) == 2:
                e1_upper, e2_upper = parts
                # 跳过双端都是参考电极的
                if e1_upper in _REF_ELECTRODES and e2_upper in _REF_ELECTRODES:
                    continue
                m1 = _resolve_single_electrode(e1_upper, standard_ch_upper, upper_to_montage)
                m2 = _resolve_single_electrode(e2_upper, standard_ch_upper, upper_to_montage)
                if m1 and m2 and m1 in pos_3d and m2 in pos_3d:
                    # 两端都找到 → 中点
                    pos = (np.array(pos_3d[m1]) + np.array(pos_3d[m2])) / 2.0
                    has_bipolar = True
                elif m1 and m1 in pos_3d and e1_upper not in _REF_ELECTRODES:
                    pos = pos_3d[m1]
                    has_bipolar = True
                elif m2 and m2 in pos_3d and e2_upper not in _REF_ELECTRODES:
                    pos = pos_3d[m2]
                    has_bipolar = True

        if pos is not None:
            valid_names.append(ch_name)
            valid_indices.append(i)
            positions_3d.append(pos)

    if len(positions_3d) == 0:
        return [], [], np.empty((0, 2)), has_bipolar

    # 3D → 2D：MNE azimuthal equidistant projection
    pts = np.array(positions_3d)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r == 0, 1, r)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    proj_r = theta / np.pi
    pos_2d = np.column_stack([proj_r * np.cos(phi), proj_r * np.sin(phi)])

    return valid_names, valid_indices, pos_2d, has_bipolar


class EEGVisualizer:
    """EEG可解释性可视化工具"""

    # 标准10-20系统电极位置（用于topomap）
    ELECTRODE_POSITIONS = {
        'Fp1': (-0.25, 0.9), 'Fp2': (0.25, 0.9),
        'F7': (-0.7, 0.55), 'F3': (-0.35, 0.55), 'Fz': (0, 0.55), 'F4': (0.35, 0.55), 'F8': (0.7, 0.55),
        'T3': (-0.9, 0), 'T7': (-0.9, 0), 'C3': (-0.45, 0), 'Cz': (0, 0), 'C4': (0.45, 0), 'T4': (0.9, 0), 'T8': (0.9, 0),
        'T5': (-0.7, -0.55), 'P7': (-0.7, -0.55), 'P3': (-0.35, -0.55), 'Pz': (0, -0.55), 'P4': (0.35, -0.55), 'T6': (0.7, -0.55), 'P8': (0.7, -0.55),
        'O1': (-0.25, -0.9), 'O2': (0.25, -0.9),
        'A1': (-1.0, 0), 'A2': (1.0, 0),
    }

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100,
                 cmap: str = 'RdBu_r'):
        """
        Args:
            figsize: 图像大小
            dpi: 图像分辨率
            cmap: 颜色映射
        """
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap

    @staticmethod
    def _plot_topomap_safe(data, pos, ax, cmap, names=None, vlim=(-1, 1)):
        """
        兼容不同 matplotlib/MNE 版本的 topomap 绘制。
        pos 可以是 mne.Info 或 ndarray (N,2)。
        降级顺序：contours=6 → contours=0（matplotlib>=3.8）→ sensors=False（overlapping）
        """
        import mne

        def _try_plot(contours, sensors, names_arg):
            return mne.viz.plot_topomap(
                data, pos, axes=ax, show=False,
                cmap=cmap, vlim=vlim,
                contours=contours, sensors=sensors,
                names=names_arg,
                image_interp='linear',
            )

        # 尝试完整渲染（带等高线）
        try:
            return _try_plot(6, True, names)
        except AttributeError:
            # matplotlib >= 3.8: QuadContourSet.collections 已移除
            pass
        except ValueError as e:
            if 'overlapping' not in str(e):
                raise

        # 降级：去掉等高线
        try:
            return _try_plot(0, True, names)
        except ValueError as e:
            if 'overlapping' not in str(e):
                raise

        # 再降级：去掉传感器点（解决 overlapping）
        return _try_plot(0, False, None)

    def plot_heatmap(self, attribution: np.ndarray, channel_names: List[str],
                     title: str = "Attribution Heatmap",
                     save_path: Optional[str] = None,
                     show: bool = True,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None) -> plt.Figure:
        """
        绘制通道×时间的热力图

        Args:
            attribution: 归因矩阵 (channels, time)
            channel_names: 通道名称
            title: 标题
            save_path: 保存路径
            show: 是否显示
            vmin, vmax: 颜色范围
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 归一化
        if vmin is None:
            vmin = np.min(attribution)
        if vmax is None:
            vmax = np.max(attribution)

        im = ax.imshow(attribution, aspect='auto', cmap=self.cmap,
                       vmin=vmin, vmax=vmax)

        # 设置坐标轴
        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names)
        ax.set_xlabel('Time (patches/samples)')
        ax.set_ylabel('Channels')
        ax.set_title(title)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"已保存: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_channel_importance(self, importance: np.ndarray, channel_names: List[str],
                                title: str = "Channel Importance",
                                top_k: Optional[int] = None,
                                save_path: Optional[str] = None,
                                show: bool = True) -> plt.Figure:
        """
        绘制通道重要性条形图

        Args:
            importance: 通道重要性 (channels,)
            channel_names: 通道名称
            title: 标题
            top_k: 只显示前k个通道
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        # 排序
        indices = np.argsort(importance)[::-1]
        if top_k:
            indices = indices[:top_k]

        sorted_importance = importance[indices]
        sorted_names = [channel_names[i] for i in indices]

        # 绘制条形图：正值红色，负值蓝色
        colors = ['#d73027' if v >= 0 else '#4575b4' for v in sorted_importance]
        bars = ax.barh(range(len(indices)), sorted_importance, color=colors)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')

        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()  # 最重要的在上面

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    # 无符号方法（值域 [0,1]），topomap 用 vlim=(0,1)
    UNSIGNED_METHODS = {'gradcam'}

    @classmethod
    def _get_vlim(cls, method_name: str) -> tuple:
        """根据方法名返回合适的 vlim"""
        if method_name.lower() in cls.UNSIGNED_METHODS:
            return (0, 1)
        return (-1, 1)

    def plot_topomap(self, importance: np.ndarray, channel_names: List[str],
                     title: str = "Topomap",
                     save_path: Optional[str] = None,
                     show: bool = True,
                     cmap: str = 'RdBu_r',
                     vlim: tuple = (-1, 1),
                     show_names: bool = True,
                     top_k: int = 5,
                     method_name: str = '',
                     metadata: Optional[dict] = None) -> plt.Figure:
        """
        绘制脑电地形图（使用MNE库生成平滑插值效果）

        Args:
            importance: 通道重要性 (channels,)
            channel_names: 通道名称
            title: 标题
            save_path: 保存路径
            show: 是否显示
            cmap: 颜色映射
            show_names: 是否显示电极名称
        """
        try:
            import mne
            return self._plot_topomap_mne(importance, channel_names, title,
                                          save_path, show, cmap, vlim, show_names, top_k,
                                          method_name, metadata)
        except ImportError:
            print("警告: MNE未安装，使用简化版topomap")
            return self._plot_topomap_simple(importance, channel_names, title,
                                             save_path, show, metadata)

    def _plot_topomap_mne(self, importance: np.ndarray, channel_names: List[str],
                          title: str, save_path: Optional[str], show: bool,
                          cmap: str = 'RdBu_r', vlim: tuple = (-1, 1),
                          show_names: bool = True,
                          top_k: int = 5,
                          method_name: str = '',
                          metadata: Optional[dict] = None) -> plt.Figure:
        """使用MNE绘制专业的脑地形图"""
        import mne

        montage = mne.channels.make_standard_montage('standard_1020')
        valid_names, valid_indices, pos_2d, has_bipolar = resolve_channel_positions(
            channel_names, montage)

        if len(valid_names) == 0:
            print("警告: 没有找到有效的标准电极，使用简化版topomap")
            return self._plot_topomap_simple(importance, channel_names, title,
                                             save_path, show, metadata)

        valid_importance = np.array([importance[i] for i in valid_indices])

        # 根据方法类型选择归一化方式和 vlim
        _vlim = self._get_vlim(method_name) if method_name else vlim
        if _vlim == (0, 1):
            # GradCAM 等无符号方法：min-max 到 [0, 1]，0→蓝，1→红
            v_min = valid_importance.min()
            v_range = valid_importance.max() - v_min + 1e-8
            norm_importance = (valid_importance - v_min) / v_range
        else:
            # 有符号方法：以 0 为中心归一化到 [-1, 1]，保留正负方向语义
            abs_max = np.abs(valid_importance).max() + 1e-8
            norm_importance = valid_importance / abs_max

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)

        # 绘制topomap - 兼容不同版本的MNE
        im, _ = self._plot_topomap_safe(
            norm_importance, pos_2d, ax, cmap,
            names=valid_names if show_names else None,
            vlim=_vlim,
        )

        # 添加colorbar（放在图下方，避免遮挡地形图）
        cbar = plt.colorbar(im, ax=ax, shrink=0.5, orientation='horizontal',
                            pad=0.05, label='Importance')

        # Top-K 标注：只取真正正/负贡献通道，正负分两行，负的从最负开始
        if top_k > 0:
            sorted_desc = np.argsort(valid_importance)[::-1]
            top_pos = [i for i in sorted_desc if valid_importance[i] > 0][:top_k]
            top_neg = [i for i in reversed(sorted_desc) if valid_importance[i] < 0][:top_k]

            lines = []
            if top_pos:
                pos_parts = [f'{valid_names[i]}(+{valid_importance[i]:.2f})' for i in top_pos]
                lines.append('Pos: ' + '  '.join(pos_parts))
            if top_neg:
                neg_parts = [f'{valid_names[i]}({valid_importance[i]:.2f})' for i in top_neg]
                lines.append('Neg: ' + '  '.join(neg_parts))
            if lines:
                fig.text(0.5, 0.01, '\n'.join(lines),
                         ha='center', va='bottom', fontsize=8, color='black')

        # Build title with metadata
        title_parts = [title]
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
                title_parts.append(' | '.join(meta_parts))

        ax.set_title('\n'.join(title_parts), fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"已保存: {save_path}")

        if show:
            plt.show()

        return fig

    def _plot_topomap_simple(self, importance: np.ndarray, channel_names: List[str],
                             title: str, save_path: Optional[str], show: bool,
                             metadata: Optional[dict] = None) -> plt.Figure:
        """简化版topomap（不依赖MNE）"""
        from scipy.interpolate import griddata

        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)

        # 绘制头部轮廓
        circle = plt.Circle((0, 0), 1, fill=False, linewidth=2, color='black')
        ax.add_patch(circle)

        # 绘制鼻子
        nose_x = [-0.1, 0, 0.1]
        nose_y = [1.0, 1.15, 1.0]
        ax.plot(nose_x, nose_y, 'k-', linewidth=2)

        # 绘制耳朵
        ear_left = plt.Circle((-1.08, 0), 0.08, fill=False, linewidth=2, color='black')
        ear_right = plt.Circle((1.08, 0), 0.08, fill=False, linewidth=2, color='black')
        ax.add_patch(ear_left)
        ax.add_patch(ear_right)

        # 收集有效电极位置和重要性
        positions = []
        values = []
        names_to_plot = []

        for i, name in enumerate(channel_names):
            name_upper = name.upper()
            pos = None
            for key in self.ELECTRODE_POSITIONS:
                if key.upper() == name_upper:
                    pos = self.ELECTRODE_POSITIONS[key]
                    break

            if pos is not None:
                positions.append(pos)
                values.append(importance[i])
                names_to_plot.append(name)

        if len(positions) < 3:
            # 位置太少，无法插值，只画点
            for pos, val, name in zip(positions, values, names_to_plot):
                norm_val = (val - min(values)) / (max(values) - min(values) + 1e-8)
                color = plt.cm.RdBu_r(norm_val)
                ax.scatter(pos[0], pos[1], c=[color], s=300, edgecolors='black', linewidths=1, zorder=3)
                ax.annotate(name, pos, fontsize=10, ha='center', va='bottom',
                           xytext=(0, 10), textcoords='offset points', fontweight='bold')
        else:
            positions = np.array(positions)
            values = np.array(values)

            # 归一化
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-8)

            # 创建插值网格
            xi = np.linspace(-1.1, 1.1, 100)
            yi = np.linspace(-1.1, 1.1, 100)
            xi, yi = np.meshgrid(xi, yi)

            # 插值
            zi = griddata(positions, norm_values, (xi, yi), method='cubic', fill_value=0)

            # 创建圆形mask
            mask = xi**2 + yi**2 > 1
            zi[mask] = np.nan

            # 绘制等高线填充图
            levels = np.linspace(0, 1, 20)
            contour = ax.contourf(xi, yi, zi, levels=levels, cmap='RdBu_r', alpha=0.8)
            ax.contour(xi, yi, zi, levels=6, colors='black', linewidths=0.5, alpha=0.5)

            # 绘制电极点和名称
            for pos, val, name in zip(positions, norm_values, names_to_plot):
                ax.scatter(pos[0], pos[1], c='black', s=30, zorder=5)
                ax.annotate(name, pos, fontsize=9, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points', fontweight='bold', zorder=6)

            # 添加colorbar
            cbar = plt.colorbar(contour, ax=ax, shrink=0.6, label='Importance')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.2, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

        # Build title with metadata
        title_parts = [title]
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
                title_parts.append(' | '.join(meta_parts))

        ax.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"已保存: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_waveform_with_attribution(self, signal: np.ndarray, attribution: np.ndarray,
                                       channel_names: List[str],
                                       sfreq: float = 200,
                                       title: str = "EEG Waveform with Attribution",
                                       channels_per_fig: int = 6,
                                       save_path: Optional[str] = None,
                                       show: bool = True) -> List[plt.Figure]:
        """
        绘制带有归因着色的波形图

        Args:
            signal: EEG信号 (channels, time)
            attribution: 归因值 (channels, time)
            channel_names: 通道名称
            sfreq: 采样率
            title: 标题
            channels_per_fig: 每张图显示的通道数
            save_path: 保存路径（会自动添加序号）
            show: 是否显示
        """
        n_channels = signal.shape[0]
        n_samples = signal.shape[1]
        time = np.arange(n_samples) / sfreq

        # 归一化归因到0-1
        attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

        figs = []
        n_figs = (n_channels + channels_per_fig - 1) // channels_per_fig

        for fig_idx in range(n_figs):
            start_ch = fig_idx * channels_per_fig
            end_ch = min(start_ch + channels_per_fig, n_channels)
            n_ch = end_ch - start_ch

            fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2 * n_ch), dpi=self.dpi)
            if n_ch == 1:
                axes = [axes]

            for i, ch_idx in enumerate(range(start_ch, end_ch)):
                ax = axes[i]

                # 绘制波形
                ax.plot(time, signal[ch_idx], 'k-', linewidth=0.5, alpha=0.7)

                # 使用归因值着色背景
                for j in range(n_samples - 1):
                    color = plt.cm.RdBu_r(attr_norm[ch_idx, j])
                    ax.axvspan(time[j], time[j + 1], alpha=0.3, color=color)

                ax.set_ylabel(channel_names[ch_idx])
                ax.set_xlim(time[0], time[-1])

                if i == n_ch - 1:
                    ax.set_xlabel('Time (s)')
                else:
                    ax.set_xticklabels([])

            fig.suptitle(f"{title} (Channels {start_ch + 1}-{end_ch})")
            plt.tight_layout()

            if save_path:
                base, ext = os.path.splitext(save_path)
                fig_path = f"{base}_ch{start_ch + 1}-{end_ch}{ext}"
                fig.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')

            if show:
                plt.show()

            figs.append(fig)

        return figs

    def plot_method_comparison(self, results: Dict[str, np.ndarray],
                               channel_names: List[str],
                               title: str = "Method Comparison",
                               save_path: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        比较多种方法的结果

        Args:
            results: {method_name: attribution_matrix} 字典
            channel_names: 通道名称
            title: 标题
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6), dpi=self.dpi)

        if n_methods == 1:
            axes = [axes]

        for ax, (method_name, attr) in zip(axes, results.items()):
            # 归一化到0-1
            attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

            im = ax.imshow(attr_norm, aspect='auto', cmap=self.cmap, vmin=0, vmax=1)
            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels(channel_names, fontsize=8)
            ax.set_xlabel('Time')
            ax.set_title(method_name)

        plt.colorbar(im, ax=axes, label='Normalized Attribution')
        fig.suptitle(title)

        # 安全地应用 tight_layout，避免警告
        try:
            plt.tight_layout()
        except Exception:
            pass

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_temporal_importance(self, results: Dict[str, np.ndarray],
                                 title: str = "Temporal Importance",
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> plt.Figure:
        """
        绘制时间维度重要性曲线

        Args:
            results: {method_name: temporal_importance} 字典
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for (method_name, importance), color in zip(results.items(), colors):
            # 归一化
            importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            ax.plot(importance_norm, label=method_name, color=color, linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Importance')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    @staticmethod
    def normalize_attribution(attribution: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        归一化归因值

        Args:
            attribution: 归因矩阵
            method: 归一化方法 ('minmax', 'zscore', 'abs_max')
        """
        if method == 'minmax':
            return (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        elif method == 'zscore':
            return (attribution - attribution.mean()) / (attribution.std() + 1e-8)
        elif method == 'abs_max':
            return attribution / (np.abs(attribution).max() + 1e-8)
        else:
            raise ValueError(f"未知的归一化方法: {method}")

    def plot_waveform_with_heatmap(self, waveform: np.ndarray, attribution: np.ndarray,
                                   channel_names: List[str],
                                   title: str = "Waveform with Attribution",
                                   save_path: Optional[str] = None,
                                   show: bool = True,
                                   high_pct: float = 95.0,
                                   low_pct: float = 5.0,
                                   spatial_importance: Optional[np.ndarray] = None) -> plt.Figure:
        """
        绘制原始波形，只显示有高归因 patch 的通道，红色标注高贡献区域。

        Args:
            waveform: (batch, channels, patches, features) 或 (channels, patches, features)
            attribution: (channels, patches)
            channel_names: 通道名称
            title: 标题
            save_path: 保存路径
            show: 是否显示
            high_pct: 高归因阈值百分位（全局），默认 95
        """
        from matplotlib.collections import LineCollection as LC
        from matplotlib.colors import Normalize

        # ── 处理输入形状 ──
        if waveform.ndim == 4:
            waveform = waveform.squeeze(0)
        if waveform.ndim == 2:
            # (channels, time) -> 当作 (channels, n_patches=time, 1)
            waveform = waveform[:, :, np.newaxis]
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform shape (channels, patches, features), got {waveform.shape}")

        n_channels, n_patches, n_features = waveform.shape
        total_points = n_patches * n_features
        waveforms_flat = waveform.reshape(n_channels, total_points)

        # attribution: 确保为 (channels, n_patches)
        if attribution.ndim == 2 and attribution.shape[1] == total_points:
            attribution = attribution.reshape(n_channels, n_patches, n_features).mean(axis=-1)
        if attribution.shape != (n_channels, n_patches):
            # shape 不匹配时插值到 n_patches
            from scipy.ndimage import zoom
            if attribution.ndim == 2 and attribution.shape[0] == n_channels:
                scale = n_patches / attribution.shape[1]
                attribution = zoom(attribution, (1, scale), order=1)
            else:
                raise ValueError(f"Unexpected attribution shape: {attribution.shape}")

        # ── 全局阈值（只用高归因） ──
        all_vals = attribution.flatten()
        v_high = np.percentile(all_vals, high_pct)

        # ── 只保留有红色 patch 的通道 ──
        ch_has_high = [ch for ch in range(n_channels) if attribution[ch].max() >= v_high]
        if len(ch_has_high) == 0:
            ch_has_high = list(range(min(n_channels, 5)))
        waveforms_flat = waveforms_flat[ch_has_high]
        attribution    = attribution[ch_has_high]
        channel_names  = [channel_names[i] for i in ch_has_high]
        n_channels     = len(ch_has_high)

        # ── 通道排序：按归因均值从大到小 ──
        if spatial_importance is not None and len(spatial_importance) >= max(ch_has_high) + 1:
            ch_importance = spatial_importance[ch_has_high]
        else:
            ch_importance = attribution.mean(axis=-1)
        sorted_desc = np.argsort(ch_importance)[::-1]
        waveforms_flat = waveforms_flat[sorted_desc]
        attribution    = attribution[sorted_desc]
        channel_names  = [channel_names[i] for i in sorted_desc]

        # ── colormap：高归因 → 红 ──
        norm_high = Normalize(vmin=v_high, vmax=all_vals.max() + 1e-8)
        cmap_high = plt.cm.Reds(np.linspace(0.45, 1.0, 256))
        cmap_high = plt.matplotlib.colors.LinearSegmentedColormap.from_list('Reds_sat', cmap_high)

        # 创建图形（按 paper 单栏宽度 ~7in 设计，1:1 嵌入时字号约 11pt）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        fig, axes = plt.subplots(n_channels, 1, figsize=(8, max(n_channels * 0.9, 1.8)), sharex=True)
        if n_channels == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.10, hspace=0.12)

        time_axis = np.arange(total_points)

        for ch in range(n_channels):
            ax = axes[ch]
            sig = waveforms_flat[ch]
            y_min, y_max = sig.min(), sig.max()
            y_pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.1

            # ── 1. 先画整段灰色波形作为底层 ──
            ax.plot(time_axis, sig, color='#aaaaaa', linewidth=0.6, zorder=2)

            # ── 2. 逐 patch 处理（只画高归因红色） ──
            for p in range(n_patches):
                val = attribution[ch, p]
                x_start = p * n_features
                x_end   = (p + 1) * n_features
                t_seg   = time_axis[x_start:x_end]
                s_seg   = sig[x_start:x_end]

                if val < v_high:
                    continue

                rgba = cmap_high(norm_high(val))
                points = np.array([t_seg, s_seg]).T.reshape(-1, 1, 2)
                segs   = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LC(segs, colors=[rgba] * len(segs), linewidths=2.0, zorder=3)
                ax.add_collection(lc)

            # ── 3. 坐标轴 ──
            ax.set_xlim(0, total_points)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ch_name = channel_names[ch] if ch < len(channel_names) else f'Ch{ch}'
            ax.set_ylabel(ch_name, fontsize=11, rotation=0, ha='right', va='center')
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


            # patch 分隔线
            for p in range(1, n_patches):
                ax.axvline(x=p * n_features, color='#cccccc', linewidth=0.4, zorder=0)

        axes[-1].set_xlabel('Time Points', fontsize=11)
        axes[-1].tick_params(axis='x', labelsize=10)
        fig.suptitle(title, fontsize=12, y=0.998)

        # ── colorbar（高归因红）──
        fig.subplots_adjust(right=0.88)
        sm_h = plt.cm.ScalarMappable(cmap=cmap_high, norm=norm_high)
        sm_h.set_array([])
        cbar_ax_h = fig.add_axes([0.905, 0.15, 0.018, 0.7])
        cb_h = fig.colorbar(sm_h, cax=cbar_ax_h)
        cb_h.set_label(f'Attribution (top {100 - int(high_pct)}%)', fontsize=10)
        cb_h.ax.tick_params(labelsize=9)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved waveform visualization to {save_path}")
            # 同时保存矢量 PDF（用于论文嵌入）
            pdf_path = os.path.splitext(save_path)[0] + '.pdf'
            fig.savefig(pdf_path, bbox_inches='tight')
            print(f"  Saved waveform PDF to {pdf_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_spatial_importance_comparison(self, results: Dict[str, np.ndarray],
                                          channel_names: List[str],
                                          title: str = "Channel Importance Comparison",
                                          save_path: Optional[str] = None,
                                          show: bool = True,
                                          use_rank: bool = True) -> plt.Figure:
        """
        绘制多个方法的通道重要性对比（曲线叠加）

        Args:
            results: {method_name: spatial_importance} 字典
            channel_names: 通道名称
            title: 标题
            save_path: 保存路径
            show: 是否显示
            use_rank: 是否使用排名（True）而不是归一化分数（False）
        """
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        x = np.arange(len(channel_names))
        n_channels = len(channel_names)

        for (method_name, importance), color in zip(results.items(), colors):
            if use_rank:
                # 转换为排名：最重要的=n_channels，最不重要的=1
                # argsort 返回从小到大的索引，argsort 的 argsort 得到排名
                ranks = np.zeros_like(importance)
                sorted_indices = np.argsort(importance)  # 从小到大
                for rank, idx in enumerate(sorted_indices):
                    ranks[idx] = rank + 1  # 排名从1开始，最小值排名1

                # 反转：让最重要的在最上面
                ranks = n_channels - ranks + 1  # 现在最重要的=n_channels，最不重要的=1

                ax.plot(x, ranks, label=method_name.upper(), color=color, linewidth=2, marker='o', markersize=4)
            else:
                # 原来的归一化方法
                importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
                ax.plot(x, importance_norm, label=method_name.upper(), color=color, linewidth=2, marker='o', markersize=4)

        ax.set_xlabel('EEG Channels', fontsize=12)

        if use_rank:
            ax.set_ylabel('Importance Rank', fontsize=12)
            ax.set_ylim(0, n_channels + 1)
            # 添加排名参考线
            ax.axhline(y=n_channels, color='gray', linestyle='--', linewidth=0.5, alpha=0.3, label='Rank 1 (Most Important)')
            ax.axhline(y=n_channels * 0.75, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.axhline(y=n_channels * 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.axhline(y=n_channels * 0.25, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.text(len(channel_names) - 0.5, n_channels - 0.5, 'Rank 1\n(Most)', fontsize=8, ha='right', va='top', color='gray')
            ax.text(len(channel_names) - 0.5, 1.5, f'Rank {n_channels}\n(Least)', fontsize=8, ha='right', va='bottom', color='gray')
        else:
            ax.set_ylabel('Normalized Importance', fontsize=12)
            ax.set_ylim(-0.05, 1.05)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_waveform_overlay_comparison(self, signal: np.ndarray,
                                          results: Dict[str, np.ndarray],
                                          channel_names: List[str],
                                          sfreq: float = 200,
                                          title: str = "Waveform Overlay Comparison",
                                          channels_per_fig: int = 4,
                                          save_path: Optional[str] = None,
                                          show: bool = True) -> List[plt.Figure]:
        """
        绘制多个方法的波形叠加对比图

        Args:
            signal: EEG信号 (channels, time)
            results: {method_name: attribution_matrix} 字典
            channel_names: 通道名称
            sfreq: 采样率
            title: 标题
            channels_per_fig: 每张图显示的通道数
            save_path: 保存路径
            show: 是否显示
        """
        from matplotlib.collections import LineCollection

        n_channels = signal.shape[0]
        n_samples = signal.shape[1]
        time = np.arange(n_samples) / sfreq
        n_methods = len(results)

        figs = []
        n_figs = (n_channels + channels_per_fig - 1) // channels_per_fig

        for fig_idx in range(n_figs):
            start_ch = fig_idx * channels_per_fig
            end_ch = min(start_ch + channels_per_fig, n_channels)
            n_ch = end_ch - start_ch

            # 创建 n_ch 行 × n_methods 列的子图
            fig, axes = plt.subplots(n_ch, n_methods,
                                    figsize=(5 * n_methods, 2.5 * n_ch),
                                    dpi=self.dpi)

            if n_ch == 1 and n_methods == 1:
                axes = np.array([[axes]])
            elif n_ch == 1:
                axes = axes.reshape(1, -1)
            elif n_methods == 1:
                axes = axes.reshape(-1, 1)

            for ch_i, ch_idx in enumerate(range(start_ch, end_ch)):
                for method_i, (method_name, attr) in enumerate(results.items()):
                    ax = axes[ch_i, method_i]

                    # 归一化归因到 0-1
                    attr_norm = (attr[ch_idx] - attr[ch_idx].min()) / \
                                (attr[ch_idx].max() - attr[ch_idx].min() + 1e-8)

                    # 创建线段和颜色
                    points = np.array([time, signal[ch_idx]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # 根据归因值生成颜色（使用RdBu_r colormap）
                    colors = plt.cm.RdBu_r(attr_norm[:-1])

                    # 根据归因值调整透明度 (0.2-1.0)
                    alphas = 0.2 + 0.8 * attr_norm[:-1]
                    colors[:, 3] = alphas

                    # 使用LineCollection绘制彩色波形
                    lc = LineCollection(segments, colors=colors, linewidths=1.5)
                    ax.add_collection(lc)

                    ax.set_xlim(time[0], time[-1])
                    ax.set_ylim(signal[ch_idx].min(), signal[ch_idx].max())

                    # 设置标签
                    if ch_i == 0:
                        ax.set_title(method_name, fontsize=11, fontweight='bold')
                    if method_i == 0:
                        ax.set_ylabel(channel_names[ch_idx], fontsize=10)
                    if ch_i == n_ch - 1:
                        ax.set_xlabel('Time (s)', fontsize=9)

                    ax.grid(alpha=0.3)

            fig.suptitle(f"{title} (Channels {start_ch + 1}-{end_ch})",
                        fontsize=13, fontweight='bold')
            plt.tight_layout()

            if save_path:
                base, ext = os.path.splitext(save_path)
                fig_path = f"{base}_ch{start_ch + 1}-{end_ch}{ext}"
                fig.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                print(f"已保存: {fig_path}")

            if show:
                plt.show()

            figs.append(fig)

        return figs

    def plot_comprehensive_comparison(self, results: Dict[str, np.ndarray],
                                      channel_names: List[str],
                                      title: str = "Comprehensive Method Comparison",
                                      save_path: Optional[str] = None,
                                      show: bool = True) -> plt.Figure:
        """
        绘制综合对比图：每个方法一列，包含时间波形、通道重要性、脑地形图

        Args:
            results: {method_name: result_dict} 字典，每个result包含:
                     - 'temporal_importance': 时间重要性
                     - 'spatial_importance': 通道重要性
            channel_names: 通道名称
            title: 标题
            save_path: 保存路径
            show: 是否显示
        """
        n_methods = len(results)

        # 创建子图：3行 × n_methods列
        # 行1: 时间波形, 行2: 通道重要性, 行3: 脑地形图
        fig = plt.figure(figsize=(5 * n_methods, 12), dpi=self.dpi)

        # 使用 GridSpec 精确控制布局
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(3, n_methods, figure=fig, hspace=0.35, wspace=0.3,
                              height_ratios=[1.2, 1.5, 2])

        method_names = list(results.keys())

        for col, method_name in enumerate(method_names):
            result = results[method_name]
            temporal_imp = result['temporal_importance']
            spatial_imp = result['spatial_importance']

            # ========== 第1行: 时间波形图 ==========
            ax_temporal = fig.add_subplot(gs[0, col])

            # 归一化
            temporal_norm = (temporal_imp - temporal_imp.min()) / (temporal_imp.max() - temporal_imp.min() + 1e-8)

            time_axis = np.arange(len(temporal_imp))
            ax_temporal.fill_between(time_axis, 0, temporal_norm, alpha=0.6, color='steelblue')
            ax_temporal.plot(time_axis, temporal_norm, color='darkblue', linewidth=2)

            ax_temporal.set_xlim(0, len(temporal_imp))
            ax_temporal.set_ylim(0, 1.05)
            ax_temporal.set_ylabel('Importance', fontsize=9)
            ax_temporal.set_title(method_name.upper(), fontsize=11, fontweight='bold')
            ax_temporal.grid(alpha=0.3, linestyle='--')

            if col == 0:
                ax_temporal.set_ylabel('Temporal\nImportance', fontsize=10, fontweight='bold')

            # ========== 第2行: 通道重要性柱状图 ==========
            ax_spatial = fig.add_subplot(gs[1, col])

            # 归一化
            spatial_norm = (spatial_imp - spatial_imp.min()) / (spatial_imp.max() - spatial_imp.min() + 1e-8)

            colors = plt.cm.YlOrRd(spatial_norm)
            bars = ax_spatial.barh(range(len(channel_names)), spatial_norm, color=colors, edgecolor='black', linewidth=0.5)

            ax_spatial.set_yticks(range(len(channel_names)))
            if col == 0:
                ax_spatial.set_yticklabels(channel_names, fontsize=8)
                ax_spatial.set_ylabel('Channel\nImportance', fontsize=10, fontweight='bold')
            else:
                ax_spatial.set_yticklabels([])

            ax_spatial.set_xlim(0, 1.05)
            ax_spatial.set_xlabel('Importance', fontsize=9)
            ax_spatial.invert_yaxis()
            ax_spatial.grid(axis='x', alpha=0.3, linestyle='--')

            # ========== 第3行: 脑地形图 ==========
            ax_topo = fig.add_subplot(gs[2, col])

            try:
                import mne

                montage = mne.channels.make_standard_montage('standard_1020')
                v_names, v_indices, v_pos2d, _ = resolve_channel_positions(
                    channel_names, montage)

                if len(v_names) >= 3:
                    valid_importance = np.array([spatial_imp[i] for i in v_indices])
                    _vlim = self._get_vlim(method_name)
                    if _vlim == (0, 1):
                        v_min = valid_importance.min()
                        valid_importance = (valid_importance - v_min) / (valid_importance.max() - v_min + 1e-8)
                    else:
                        abs_max = np.abs(valid_importance).max() + 1e-8
                        valid_importance = valid_importance / abs_max

                    im, _ = self._plot_topomap_safe(
                        valid_importance, v_pos2d, ax_topo, 'RdBu_r', names=None, vlim=_vlim,
                    )

                    if col == 0:
                        ax_topo.text(-0.15, 0.5, 'Topomap', transform=ax_topo.transAxes,
                                   fontsize=10, fontweight='bold', rotation=90,
                                   ha='center', va='center')
                else:
                    ax_topo.text(0.5, 0.5, 'Not enough\nvalid channels',
                               ha='center', va='center', transform=ax_topo.transAxes, fontsize=10)
                    ax_topo.axis('off')

            except ImportError:
                ax_topo.text(0.5, 0.5, 'MNE not\ninstalled',
                           ha='center', va='center', transform=ax_topo.transAxes, fontsize=10)
                ax_topo.axis('off')

        # 设置总标题
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # 保存
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved comprehensive comparison to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_topomap_comparison(self, results: Dict[str, np.ndarray],
                               channel_names: List[str],
                               title: str = "Topomap Comparison",
                               save_path: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        绘制多个方法的脑地形图对比（横向排列）

        Args:
            results: {method_name: spatial_importance} 字典
            channel_names: 通道名称
            title: 标题
            save_path: 保存路径
            show: 是否显示
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6), dpi=self.dpi)

        if n_methods == 1:
            axes = [axes]

        try:
            import mne
            montage = mne.channels.make_standard_montage('standard_1020')

            for ax, (method_name, importance) in zip(axes, results.items()):
                v_names, v_indices, v_pos2d, _ = resolve_channel_positions(
                    channel_names, montage)

                if len(v_names) < 3:
                    ax.text(0.5, 0.5, f'{method_name.upper()}\n(Not enough channels)',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.axis('off')
                    continue

                valid_importance = np.array([importance[i] for i in v_indices])
                _vlim = self._get_vlim(method_name)
                if _vlim == (0, 1):
                    v_min = valid_importance.min()
                    norm_importance = (valid_importance - v_min) / (valid_importance.max() - v_min + 1e-8)
                else:
                    abs_max = np.abs(valid_importance).max() + 1e-8
                    norm_importance = valid_importance / abs_max

                im, _ = self._plot_topomap_safe(
                    norm_importance, v_pos2d, ax, 'RdBu_r', names=v_names, vlim=_vlim,
                )

                ax.set_title(method_name.upper(), fontsize=12, fontweight='bold')

            # 添加共享的 colorbar
            if n_methods > 0:
                cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.6, label='Importance')

        except ImportError:
            # 如果 MNE 不可用，显示警告
            for ax, (method_name, _) in zip(axes, results.items()):
                ax.text(0.5, 0.5, f'{method_name.upper()}\n(MNE required)',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # MNE topomap 与 tight_layout 不兼容，使用 try-except 忽略警告
        try:
            plt.tight_layout()
        except Exception:
            pass  # MNE topomap 创建的 axes 可能不兼容 tight_layout

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

