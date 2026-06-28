"""
Population-level LLM 解读（不重算归因，直接读已有输出）。

输入: 一个 population 输出目录，包含 topomap_data.json 和若干 PNG。
输出: 同目录下生成 llm_interpretation.json / .txt。

默认: picture 模式 + Claude (claude-sonnet-4-6)，与 explainability/llm_interpret.py 一致。

用法:
    python -m explainability.llm_interpret_population \
        --result-dir population_results/population/stress/cbramod_xxx/deeplift/class_0_TP \
        --task stress \
        --model-type cbramod
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def _load_llm_interpret_module():
    """直接按文件路径加载 llm_interpret.py，绕过 explainability/__init__.py
    （后者会触发 torch/captum/shap/lime 等重型依赖，启动慢 10-30s）。"""
    here = Path(__file__).resolve().parent
    target = here / "llm_interpret.py"
    spec = importlib.util.spec_from_file_location("_llm_interpret_standalone", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_llm_interpret_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod


_li = _load_llm_interpret_module()
_channel_region = _li._channel_region
_channel_laterality = _li._channel_laterality
_encode_image = _li._encode_image
_query_claude_vision = _li._query_claude_vision
_query_openai_vision = _li._query_openai_vision
_query_text_only = _li._query_text_only


SYSTEM_PROMPT_POPULATION_ZH = """\
你是一位神经科学与脑电信号分析专家，精通深度学习可解释性方法（XAI）。

## ⚠️ 核心概念（必须理解）

你看到的是**群体层面（population-level）的模型归因结果**，即 N 个样本（同一目标类别、同一预测结果类型）归因图的**平均**。

- 归因值 > 0：该特征在群体中**平均地**支持模型做出当前类别预测
- 归因值 < 0：该特征在群体中**平均地**抑制当前类别预测
- 这**不是**脑电信号的原始功率或幅值，不能解读为"该脑区活动增强/减弱"
- 群体平均会**抵消个体差异**：稳定出现的模式会被保留，零散个体特征会被平滑掉
- **TP vs FP** 区分：TP 反映模型"答对时依据什么"，FP 反映"答错时被什么误导"，解读侧重不同

## 图片说明

- **Grand-Average Spatial Topomap（群体平均空间地形图）**：群体中各通道对预测的平均贡献。红=平均正向支持，蓝=平均反向抑制。颜色越深说明群体一致性越高。
- **Grand-Average Temporal Waveform（群体平均时间归因）**：各时间段/patch 在群体中的平均贡献权重（仅 event 任务通常会有）。
- **Band Topomap（频段地形图）**：各频段 × 通道在群体上的平均归因。频段：Delta(0.5-4Hz), Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz), Gamma(30-45Hz)。

## 你的任务

综合空间、（如有）时间、频段三个维度，结合**任务的神经生理学先验**（数据集描述与类别定义已给出），评估群体层面归因模式是否合理：

1. 群体一致出现的脑区/频段/时段是否符合该任务的已知神经标志？
2. 若分析的是 FP 样本，模型是否被某种**伪迹**或**任务无关特征**系统性误导？
3. 偏侧性（左右半球）模式是否与任务对称性预期一致？

## 输出格式（严格遵守）

**判定：**[群体归因整体是否合理] 可信度：[高/中/低] 样本规模：[基于 n_samples 评论统计稳定性]

**核心发现：**

1. [第一条发现：综合相关维度，说明某个群体层面的归因模式及其神经生理学意义。结尾标注 ✅（合理）/ ⚠️（需注意）/ ❌（不合理）]

2. [第二条发现：同上]

3. [第三条发现：同上，如有必要]

**总评：**[2-3 句话，概括群体模式哪些有文献支撑、哪些可能反映过拟合或数据偏差，给出整体可解释性评价]

## 写作要求

- 每条发现是一个完整的群体洞察，自然融合空间/时间/频段，不要机械按维度拆分
- 用"群体中""平均而言""多数样本"等措辞，避免暗示这是单样本结论
- 合理与可疑的发现都要有（不只说好话也不只挑刺）
- 引用关键通道/频段/时段作为证据，但不罗列数值表格
- 不确定的内容用"可能""推测"标注
- 如怀疑系统性伪迹（如多数样本前额电极主导可能是眼电），直接在相关发现中说明

## 约束

- 总输出 300-500 字，中文，简洁直白
- 不做临床诊断
- 不要重复描述图中能直接看到的数值，重点放在群体层面的解读与判断
- 区分 TP/FP 时，明确指出当前是哪种以及对应的解读侧重
"""


def _load_yaml_config(model_type: str, task: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "configs" / f"{model_type}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = data["CONFIGS"][task]
    if "task_type" not in cfg:
        spec2 = importlib.util.spec_from_file_location(
            "_task_configs_standalone",
            Path(__file__).resolve().parent / "task_configs.py",
        )
        tc_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(tc_mod)
        if task.lower() in tc_mod.TASK_CONFIGS:
            cfg["task_type"] = tc_mod.TASK_CONFIGS[task.lower()].task_type
    return cfg


def _build_population_context(
    meta: Dict[str, Any],
    channel_importance: Dict[str, float],
    config: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    channels = list(channel_importance.keys())
    importance = np.array([channel_importance[c] for c in channels], dtype=float)

    top_idx = np.argsort(np.abs(importance))[::-1][:top_k]
    region_imp: Dict[str, float] = {}
    lat_imp: Dict[str, float] = {"Left": 0.0, "Right": 0.0, "Midline": 0.0}
    for ch, val in zip(channels, importance):
        region_imp[_channel_region(ch)] = region_imp.get(_channel_region(ch), 0.0) + float(val)
        lat_imp[_channel_laterality(ch)] += float(val)
    total_r = sum(abs(v) for v in region_imp.values()) or 1.0
    total_l = sum(abs(v) for v in lat_imp.values()) or 1.0

    class_names = config.get("class_names", {}) or {}
    target_class = meta.get("target_class")

    ctx: Dict[str, Any] = {
        "analysis_type": "population",
        "model": meta.get("model", ""),
        "method": meta.get("method", ""),
        "task_type": config.get("task_type", "state"),
        "dataset_description": config.get("dataset_description", ""),
        "class_definitions": {str(k): v for k, v in class_names.items()},
        "population_meta": {
            "n_samples": meta.get("n_samples"),
            "sample_type": meta.get("sample_type"),
            "target_class": target_class,
            "target_class_name": class_names.get(target_class, class_names.get(str(target_class), f"Class {target_class}")),
            "conf_threshold": meta.get("conf_threshold"),
        },
        "spatial": {
            "description": "红色(正值)=正向支持该类别预测, 蓝色(负值)=反向抑制该类别预测",
            "top_channels": [
                {
                    "rank": i + 1,
                    "channel": channels[idx],
                    "importance": round(float(importance[idx]), 4),
                    "direction": "正向支持" if importance[idx] > 0 else "反向抑制",
                    "region": _channel_region(channels[idx]),
                    "laterality": _channel_laterality(channels[idx]),
                }
                for i, idx in enumerate(top_idx)
            ],
            "region_distribution_pct": {k: round(v / total_r * 100, 1) for k, v in region_imp.items()},
            "laterality_pct": {k: round(v / total_l * 100, 1) for k, v in lat_imp.items()},
            "dominant_region": max(region_imp, key=lambda k: abs(region_imp[k])),
        },
    }
    return ctx


def _collect_population_images(result_dir: str) -> List[Dict[str, str]]:
    images: List[Dict[str, str]] = []
    seen = set()

    def add(path: str, label: str):
        ap = os.path.abspath(path)
        if ap in seen or not os.path.exists(path):
            return
        seen.add(ap)
        images.append({"path": path, "label": label})

    # 根目录下常见图
    add(os.path.join(result_dir, "grand_avg_topomap.png"),
        "Grand-Average Spatial Topomap (群体平均空间地形图): 红=正向支持, 蓝=反向抑制")
    add(os.path.join(result_dir, "grand_avg_waveform.png"),
        "Grand-Average Temporal Waveform (群体平均时间波形): 各时段的归因重要性")

    # 根目录及子目录里的 band/topomap/waveform PNG
    patterns = ["band_topomap*.png", "topomap*.png", "waveform*.png", "*topomap*.png", "*waveform*.png"]
    search_dirs = [result_dir,
                   os.path.join(result_dir, "band_topomap"),
                   os.path.join(result_dir, "band_attribution"),
                   os.path.join(result_dir, "bands")]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            for p in sorted(glob.glob(os.path.join(d, pat))):
                name = os.path.basename(p)
                sub = os.path.basename(d) if d != result_dir else ""
                label = f"{'['+sub+'] ' if sub else ''}{name}"
                add(p, label)

    return images


def _load_extra_json(result_dir: str) -> Dict[str, Any]:
    """读取 population_summary.json 与 band_attribution/*.json，归并进 context。"""
    extra: Dict[str, Any] = {}

    summary_path = os.path.join(result_dir, "population_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            extra["population_summary"] = json.load(f)

    band_dir = os.path.join(result_dir, "band_attribution")
    if os.path.isdir(band_dir):
        band_jsons = sorted(glob.glob(os.path.join(band_dir, "*.json")))
        if band_jsons:
            extra["band_attribution"] = {}
            for jp in band_jsons:
                with open(jp, "r", encoding="utf-8") as f:
                    extra["band_attribution"][os.path.basename(jp)] = json.load(f)

    return extra


def _build_user_text(ctx: Dict[str, Any]) -> str:
    m = ctx["population_meta"]
    lines = [
        "## 分析背景 (Population-level)",
        f"- 数据集: {ctx.get('dataset_description', 'N/A')}",
        f"- 模型: {ctx['model']}",
        f"- 归因方法: {ctx['method']}",
        f"- 任务类型: {ctx['task_type']}",
        f"- 类别定义: {json.dumps(ctx.get('class_definitions', {}), ensure_ascii=False)}",
        "",
        "## 群体信息",
        f"- 目标类别: {m['target_class']} ({m['target_class_name']})",
        f"- 样本筛选: {m['sample_type']} (置信度阈值 {m['conf_threshold']})",
        f"- 样本数: {m['n_samples']}",
        "",
        "## 归因数据 (群体平均)",
        "```json",
        json.dumps({k: v for k, v in ctx.items() if k in ("spatial", "temporal", "band", "population_summary", "band_attribution")},
                   ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    return "\n".join(lines)


def interpret_population(
    result_dir: str,
    task: str,
    model_type: str,
    mode: str = "picture",
    llm: str = "claude",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    max_tokens: int = 2000,
    top_k: int = 10,
) -> Dict[str, Any]:
    json_path = os.path.join(result_dir, "topomap_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"topomap_data.json not found in: {result_dir}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = dict(data.get("meta", {}))
    meta["model"] = model_type
    channel_importance = data["channel_importance"]

    config = _load_yaml_config(model_type, task)
    ctx = _build_population_context(meta, channel_importance, config, top_k=top_k)
    extra = _load_extra_json(result_dir)
    if extra:
        ctx.update(extra)
        print(f"  [Info] Loaded extra: {list(extra.keys())}")
    user_text = _build_user_text(ctx)

    print(f"\n{'='*60}\n[Population LLM Interpret] mode={mode}, provider={llm}\n{'='*60}")

    if mode == "picture":
        images = _collect_population_images(result_dir)
        if not images:
            print("  [Warning] No images found, falling back to json mode.")
            mode = "json"

    if mode == "picture":
        print(f"  Sending {len(images)} images + context to {llm}...")
        if llm == "claude":
            interpretation = _query_claude_vision(SYSTEM_PROMPT_POPULATION_ZH, user_text, images,
                                                  model=llm_model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)
        elif llm == "openai":
            interpretation = _query_openai_vision(SYSTEM_PROMPT_POPULATION_ZH, user_text, images,
                                                  model=llm_model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)
        else:
            print(f"  [Warning] {llm} 不支持 vision, 回退 json 模式")
            interpretation = _query_text_only(SYSTEM_PROMPT_POPULATION_ZH, user_text, llm=llm, model=llm_model,
                                              api_key=api_key, api_base=api_base, max_tokens=max_tokens)
    else:
        print(f"  Sending structured JSON context to {llm}...")
        interpretation = _query_text_only(SYSTEM_PROMPT_POPULATION_ZH, user_text, llm=llm, model=llm_model,
                                          api_key=api_key, api_base=api_base, max_tokens=max_tokens)

    print(f"\n{'-'*60}\nLLM 群体归因解读结果:\n{'-'*60}\n{interpretation}\n{'-'*60}")

    result = {"context": ctx, "interpretation": interpretation, "mode": mode, "llm": llm, "llm_model": llm_model}
    out_json = os.path.join(result_dir, "llm_interpretation.json")
    out_txt = os.path.join(result_dir, "llm_interpretation.txt")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(interpretation)
    print(f"[Saved] {out_json}\n[Saved] {out_txt}")
    return result


def main():
    p = argparse.ArgumentParser(description="Population-level LLM interpretation (re-uses existing population output).")
    p.add_argument("--result-dir", required=True, help="目录: 含 topomap_data.json 和 PNG (如 .../deeplift/class_0_TP)")
    p.add_argument("--task", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--mode", default="picture", choices=["picture", "json"])
    p.add_argument("--llm", default="claude", choices=["claude", "openai", "deepseek"])
    p.add_argument("--llm-model", default=None, help="默认 claude-sonnet-4-6 / gpt-4o，与 llm_interpret.py 保持一致")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    interpret_population(
        result_dir=args.result_dir, task=args.task, model_type=args.model_type,
        mode=args.mode, llm=args.llm, llm_model=args.llm_model,
        api_key=args.api_key, api_base=args.api_base,
        max_tokens=args.max_tokens, top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
