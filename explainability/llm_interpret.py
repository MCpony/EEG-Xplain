"""
EEG Single-Sample LLM Interpretation Module
============================================

在 run_explainability 完成归因分析后，自动将结果发送给 LLM 进行
生理合理性判断。支持两种模式：

- picture: 将生成的 PNG 图片（topomap, waveform, band_topomap）编码为
  base64 发送给多模态 LLM（Claude / GPT-4o）
- json: 提取关键数值数据为结构化 JSON，发送给文本 LLM（含 DeepSeek）

Author: auto-generated
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ===================== 脑区映射 =====================

REGION_MAP = {
    "Frontal": {"Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "AF3", "AF4", "AF7", "AF8"},
    "Temporal": {"T3", "T4", "T5", "T6", "T7", "T8", "TP7", "TP8", "FT7", "FT8"},
    "Central": {"C3", "C4", "Cz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz"},
    "Parietal": {"P3", "P4", "Pz", "P7", "P8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz"},
    "Occipital": {"O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8"},
}

LATERALITY_LEFT = {"F7", "F3", "T3", "T7", "C3", "P3", "P7", "T5", "O1", "Fp1",
                   "FC1", "FC3", "FC5", "FT7", "CP1", "CP3", "CP5", "TP7", "PO3", "PO7", "AF3", "AF7"}
LATERALITY_RIGHT = {"F8", "F4", "T4", "T8", "C4", "P4", "P8", "T6", "O2", "Fp2",
                    "FC2", "FC4", "FC6", "FT8", "CP2", "CP4", "CP6", "TP8", "PO4", "PO8", "AF4", "AF8"}


def _channel_region(ch: str) -> str:
    ch_upper = ch.upper()
    for region, names in REGION_MAP.items():
        if ch_upper in {n.upper() for n in names}:
            return region
    return "Other"


def _channel_laterality(ch: str) -> str:
    ch_upper = ch.upper()
    if ch_upper in {n.upper() for n in LATERALITY_LEFT}:
        return "Left"
    if ch_upper in {n.upper() for n in LATERALITY_RIGHT}:
        return "Right"
    return "Midline"


# ===================== System Prompts =====================

SYSTEM_PROMPT_ZH = """\
你是一位神经科学与脑电信号分析专家，精通深度学习可解释性方法（XAI）。

## ⚠️ 核心概念（必须理解）

你看到的数值和图像是**模型归因值（attribution）**，表示"模型做出该预测时，各通道/时段/频段对决策的贡献程度"。

- 归因值 > 0：该特征**支持**模型做出当前预测
- 归因值 < 0：该特征**抑制/反对**当前预测
- 这**不是**脑电信号的原始功率或幅值，不能直接解读为"该区域活动增强/减弱"

## 图片说明

- **Spatial Topomap（空间归因图）**：各通道对模型预测的贡献。红=正向支持，蓝=反向抑制。
- **Temporal Waveform（时间归因图）**：各时间段/patch对预测的贡献权重。反映模型在时间维度上的关注分布。
- **Band Topomap（频段归因图）**：各频段×通道的归因分布。频段：Delta(0.5-4Hz), Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz), Gamma(30-45Hz)。

## 你的任务

综合空间、时间、频段三个维度的归因信息，提炼出2-4条**核心发现**，评估模型决策的神经生理学合理性。每条发现可以跨维度整合，不要按维度割裂分析。

## 输出格式（严格遵守）

**判定：**[模型归因整体是否合理] 可信度：[高/中/低]

**核心发现：**

1. [第一条发现：综合相关维度的信息，说明某个归因模式及其神经生理学意义。结尾标注 ✅（合理）/ ⚠️（需注意）/ ❌（不合理）]

2. [第二条发现：同上]

3. [第三条发现：同上，如有必要]

**总评：**[2-3句话，概括哪些方面有文献支撑、哪些需要进一步验证，给出整体可解释性评价]

## 写作要求

- 每条发现是一个完整的洞察，自然融合空间/时间/频段信息，不要机械地按维度拆分
- 合理的发现和可疑的发现都要有，比例大致均衡（不要只说好话，也不要只挑刺）
- 引用关键通道/频段/时段作为证据，但不要罗列数值表格
- 不确定的内容用"可能""推测"标注，不要断言
- 如发现伪迹嫌疑（肌电、眼电等），在相关发现中直接说明，不单独成节

## 约束

- 总输出控制在300-500字
- 用中文，语言简洁直白
- 不做临床诊断
- 不要重复描述图中已经能直接看到的数值，重点放在解读和判断
"""


# ===================== Context Builder =====================

def build_sample_context(
    config: Dict[str, Any],
    channel_names: List[str],
    spatial_importance: np.ndarray,
    pred_class: int,
    confidence: float,
    true_label: Optional[int] = None,
    combined_attribution: Optional[np.ndarray] = None,
    band_result_matrix: Optional[np.ndarray] = None,
    band_names: Optional[List[str]] = None,
    method: str = '',
    model_type: str = '',
    task: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    从内存中的归因结果构建结构化 context，供 LLM 解读。
    """
    task_type = config.get('task_type', 'state')
    class_names = config.get('class_names', {})
    dataset_desc = config.get('dataset_description', '')

    ctx: Dict[str, Any] = {
        "model": model_type,
        "method": method,
        "task": task or '',
        "task_type": task_type,
        "dataset_description": dataset_desc,
        "prediction": {
            "predicted_class": pred_class,
            "predicted_class_name": class_names.get(pred_class, class_names.get(str(pred_class), f"Class {pred_class}")),
            "confidence": round(confidence, 4),
        },
        "class_definitions": {
            str(k): v for k, v in class_names.items()
        } if class_names else {},
    }

    if true_label is not None:
        ctx["prediction"]["true_label"] = true_label
        ctx["prediction"]["true_label_name"] = class_names.get(true_label, class_names.get(str(true_label), f"Class {true_label}"))
        ctx["prediction"]["correct"] = (pred_class == true_label)

    # Spatial attribution
    importance = np.array(spatial_importance)
    top_idx = np.argsort(np.abs(importance))[::-1][:top_k]
    region_imp: Dict[str, float] = {}
    lat_imp: Dict[str, float] = {"Left": 0.0, "Right": 0.0, "Midline": 0.0}

    for ch, val in zip(channel_names, importance):
        r = _channel_region(ch)
        region_imp[r] = region_imp.get(r, 0.0) + float(val)
        lat_imp[_channel_laterality(ch)] += float(val)

    total_region = sum(abs(v) for v in region_imp.values()) or 1.0
    total_lat = sum(abs(v) for v in lat_imp.values()) or 1.0

    ctx["spatial"] = {
        "description": "红色(正值)=正向支持该类别预测, 蓝色(负值)=反向抑制该类别预测",
        "top_channels": [
            {
                "rank": i + 1,
                "channel": channel_names[idx],
                "importance": round(float(importance[idx]), 4),
                "direction": "正向支持" if importance[idx] > 0 else "反向抑制",
                "region": _channel_region(channel_names[idx]),
                "laterality": _channel_laterality(channel_names[idx]),
            }
            for i, idx in enumerate(top_idx)
        ],
        "region_distribution_pct": {k: round(v / total_region * 100, 1) for k, v in region_imp.items()},
        "laterality_pct": {k: round(v / total_lat * 100, 1) for k, v in lat_imp.items()},
    }

    # Temporal attribution (event tasks only) - 使用完整 (n_channels, n_patches) 矩阵
    if combined_attribution is not None:
        combined_attr = np.array(combined_attribution)
        if combined_attr.ndim == 2:
            n_ch, n_patches = combined_attr.shape
            epoch_duration = config.get('epoch_duration', n_patches)
            patch_duration = epoch_duration / n_patches if n_patches > 0 else 1.0

            # 全局找 top-K 个 (channel, patch) 对
            flat_abs = np.abs(combined_attr).flatten()
            top_flat_idx = np.argsort(flat_abs)[::-1][:top_k]

            # 按通道分组
            channel_patches = {}  # ch_idx -> list of patch info
            for flat_i in top_flat_idx:
                ch_i = int(flat_i // n_patches)
                p_i = int(flat_i % n_patches)
                if ch_i not in channel_patches:
                    channel_patches[ch_i] = []
                channel_patches[ch_i].append({
                    "patch_index": p_i,
                    "time_range": f"{p_i * patch_duration:.2f}-{(p_i + 1) * patch_duration:.2f}s",
                    "importance": round(float(combined_attr[ch_i, p_i]), 4),
                })

            ctx["temporal"] = {
                "description": "全局归因最高的(通道,时间段)组合，按通道分组展示",
                "epoch_duration_sec": epoch_duration,
                "epoch_structure": config.get('epoch_structure', ''),
                "patch_duration_sec": round(patch_duration, 2),
                "n_patches": n_patches,
                "temporal_channels": [
                    {
                        "channel": channel_names[ch_i] if ch_i < len(channel_names) else f"Ch{ch_i}",
                        "region": _channel_region(channel_names[ch_i]) if ch_i < len(channel_names) else "Unknown",
                        "laterality": _channel_laterality(channel_names[ch_i]) if ch_i < len(channel_names) else "Unknown",
                        "highlighted_patches": patches,
                    }
                    for ch_i, patches in sorted(
                        channel_patches.items(),
                        key=lambda x: max(abs(p["importance"]) for p in x[1]),
                        reverse=True,
                    )
                ],
            }

    # Band attribution
    if band_result_matrix is not None and band_names is not None:
        band_matrix = np.array(band_result_matrix)
        n_ch, n_bands = band_matrix.shape

        global_band_importance = np.mean(np.abs(band_matrix), axis=0)
        global_total = global_band_importance.sum() or 1.0

        top_pairs = []
        flat_abs = np.abs(band_matrix).flatten()
        top_flat_idx = np.argsort(flat_abs)[::-1][:top_k]
        for flat_i in top_flat_idx:
            ch_i = flat_i // n_bands
            b_i = flat_i % n_bands
            top_pairs.append({
                "channel": channel_names[ch_i] if ch_i < len(channel_names) else f"Ch{ch_i}",
                "band": band_names[b_i],
                "importance": round(float(band_matrix[ch_i, b_i]), 4),
                "direction": "正向支持" if band_matrix[ch_i, b_i] > 0 else "反向抑制",
                "region": _channel_region(channel_names[ch_i]) if ch_i < len(channel_names) else "Unknown",
            })

        ctx["band"] = {
            "description": "各频段在各通道上的归因贡献 (Delta 0.5-4Hz, Theta 4-8Hz, Alpha 8-13Hz, Beta 13-30Hz, Gamma 30-45Hz)",
            "global_band_pct": {
                band_names[i]: round(float(global_band_importance[i] / global_total * 100), 1)
                for i in range(n_bands)
            },
            "dominant_band": band_names[int(np.argmax(global_band_importance))],
            "top_channel_band_pairs": top_pairs,
        }

    return ctx


# ===================== Image Encoding =====================

def _encode_image(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _collect_images(output_dir: str, method: str) -> List[Dict[str, str]]:
    images = []
    topomap_path = os.path.join(output_dir, f"{method}_topomap.png")
    if os.path.exists(topomap_path):
        images.append({
            "path": topomap_path,
            "label": "Spatial Topomap (空间地形图): 红色=正向支持, 蓝色=反向抑制",
        })

    waveform_path = os.path.join(output_dir, f"{method}_waveform.png")
    if os.path.exists(waveform_path):
        images.append({
            "path": waveform_path,
            "label": "Temporal Waveform (时间波形图): 各时间段的归因重要性",
        })

    import glob as _glob
    band_dir = os.path.join(output_dir, "band_topomap")
    band_files = sorted(_glob.glob(os.path.join(band_dir, "band_topomap*.png")))
    for band_path in band_files:
        images.append({
            "path": band_path,
            "label": f"Band Topomap (频段地形图): {os.path.basename(band_path)}",
        })

    return images


# ===================== LLM API Calls =====================

def _build_user_message_text(ctx: Dict[str, Any]) -> str:
    pred = ctx["prediction"]
    lines = [
        "## 分析背景",
        f"- 数据集: {ctx.get('dataset_description', 'N/A')}",
        f"- 模型: {ctx['model']}",
        f"- 归因方法: {ctx['method']}",
        f"- 任务类型: {ctx['task_type']}",
        f"- 类别定义: {json.dumps(ctx.get('class_definitions', {}), ensure_ascii=False)}",
        "",
        "## 预测结果",
        f"- 预测类别: {pred['predicted_class']} ({pred['predicted_class_name']})",
        f"- 置信度: {pred['confidence']}",
    ]
    if "true_label" in pred:
        lines.append(f"- 真实类别: {pred['true_label']} ({pred['true_label_name']})")
        lines.append(f"- 预测{'正确 ✓' if pred['correct'] else '错误 ✗'}")

    lines.append("")
    lines.append("## 归因数据")
    lines.append("```json")
    data_subset = {k: v for k, v in ctx.items() if k in ("spatial", "temporal", "band")}
    lines.append(json.dumps(data_subset, ensure_ascii=False, indent=2, default=str))
    lines.append("```")

    return "\n".join(lines)


def _query_claude_vision(
    system: str, text: str, images: List[Dict[str, str]],
    model: Optional[str] = None, api_key: Optional[str] = None,
    api_base: Optional[str] = None, max_tokens: int = 2000,
) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not found. Install with: pip install anthropic")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client_kwargs = {"api_key": key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = anthropic.Anthropic(**client_kwargs)
    content = []

    for img_info in images:
        b64 = _encode_image(img_info["path"])
        if b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
            content.append({"type": "text", "text": f"[图片说明] {img_info['label']}"})

    content.append({"type": "text", "text": text})

    msg = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")


def _query_openai_vision(
    system: str, text: str, images: List[Dict[str, str]],
    model: Optional[str] = None, api_key: Optional[str] = None,
    api_base: Optional[str] = None, max_tokens: int = 2000,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not found. Install with: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set.")

    client_kwargs = {"api_key": key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)
    content = []

    for img_info in images:
        b64 = _encode_image(img_info["path"])
        if b64:
            content.append({"type": "text", "text": f"[图片说明] {img_info['label']}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

    content.append({"type": "text", "text": text})

    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    return resp.choices[0].message.content


def _query_text_only(
    system: str, text: str,
    llm: str = "claude", model: Optional[str] = None,
    api_key: Optional[str] = None, api_base: Optional[str] = None,
    max_tokens: int = 2000,
) -> str:
    if llm == "claude":
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not found. Install with: pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        client_kwargs = {"api_key": key}
        if api_base:
            client_kwargs["base_url"] = api_base
        client = anthropic.Anthropic(**client_kwargs)
        msg = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        return next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")

    elif llm == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        client_kwargs = {"api_key": key}
        if api_base:
            client_kwargs["base_url"] = api_base
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content

    elif llm == "deepseek":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
        key = api_key or os.environ.get("DEEPSEEK_API_KEY") or "sk-cebc42019aef49c9827e52ce746ac7a6"
        base = api_base or "https://api.deepseek.com"
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM provider: {llm!r}")


# ===================== High-Level Entry =====================

def interpret_sample(
    config: Dict[str, Any],
    channel_names: List[str],
    spatial_importance: np.ndarray,
    pred_class: int,
    confidence: float,
    method: str,
    model_type: str,
    output_dir: str,
    mode: str = "json",
    llm: str = "claude",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    true_label: Optional[int] = None,
    task: Optional[str] = None,
    combined_attribution: Optional[np.ndarray] = None,
    band_result_matrix: Optional[np.ndarray] = None,
    band_names: Optional[List[str]] = None,
    top_k: int = 10,
    max_tokens: int = 2000,
    save: bool = True,
) -> Dict[str, Any]:
    """
    单样本 LLM 归因解读入口。

    Args:
        mode: 'picture' (多模态，发送图片) 或 'json' (纯文本，发送结构化数据)
        llm: 'claude', 'openai', 'deepseek'
        其余参数来自 run_explainability 运行时的内存数据。

    Returns:
        {"context": dict, "interpretation": str, "mode": str, "llm": str}
    """
    print(f"\n{'='*60}")
    print(f"[LLM Interpret] mode={mode}, provider={llm}")
    print(f"{'='*60}")

    ctx = build_sample_context(
        config=config,
        channel_names=channel_names,
        spatial_importance=spatial_importance,
        pred_class=pred_class,
        confidence=confidence,
        true_label=true_label,
        combined_attribution=combined_attribution,
        band_result_matrix=band_result_matrix,
        band_names=band_names,
        method=method,
        model_type=model_type,
        task=task,
        top_k=top_k,
    )

    user_text = _build_user_message_text(ctx)
    interpretation = None

    if mode == "picture":
        images = _collect_images(output_dir, method)
        if not images:
            print("  [Warning] No images found, falling back to json mode.")
            mode = "json"

    if mode == "picture":
        print(f"  Sending {len(images)} images + context to {llm}...")
        if llm == "deepseek":
            print("  [Warning] DeepSeek does not support vision. Falling back to json mode.")
            interpretation = _query_text_only(SYSTEM_PROMPT_ZH, user_text, llm=llm, model=llm_model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)
        elif llm == "claude":
            interpretation = _query_claude_vision(SYSTEM_PROMPT_ZH, user_text, images, model=llm_model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)
        elif llm == "openai":
            interpretation = _query_openai_vision(SYSTEM_PROMPT_ZH, user_text, images, model=llm_model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported LLM for picture mode: {llm}")
    else:
        print(f"  Sending structured JSON context to {llm}...")
        interpretation = _query_text_only(SYSTEM_PROMPT_ZH, user_text, llm=llm, model=llm_model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)

    print(f"\n{'─'*60}")
    print("LLM 归因解读结果:")
    print(f"{'─'*60}")
    print(interpretation)
    print(f"{'─'*60}\n")

    result = {
        "context": ctx,
        "interpretation": interpretation,
        "mode": mode,
        "llm": llm,
        "llm_model": llm_model,
    }

    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "llm_interpretation.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Saved] {save_path}")

        txt_path = os.path.join(output_dir, "llm_interpretation.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(interpretation)
        print(f"[Saved] {txt_path}")

    return result
