# EEG-Xplain

EEG 深度学习分类模型的可解释性分析框架。支持多种 EEG 基础模型 / 多种归因方法 / 单样本与群体级分析 / LLM 自动解读。

---

## 主要特性

- **多模型支持**：CBraMod、LaBraM、EEGMamba、EEGPT、BIOT
- **多归因方法**：GradCAM、Integrated Gradients、Gradient SHAP、SHAP、LIME、Occlusion
- **多维度分析**
  - 单样本归因（通道地形图、波形归因）
  - 群体级归因（跨样本平均 + 统计检验）
  - 频段归因（Delta/Theta/Alpha/Beta/Gamma 五频段分别贡献）
  - Patch 归因 vs 频段能量相关性（Spearman ρ）
  - 忠实度评估（AOPC）
- **LLM 自动解读**：调 Claude / OpenAI / DeepSeek 生成可读报告

---

## 项目结构

```
EEG-Xplain/
├── configs/                  # 模型 YAML 配置（cbramod/labram/eegmamba/eegpt/biot）
├── model_list/               # 模型定义
├── pretrained-models/        # 预训练权重（需自备）
├── data/                     # 数据（不上传 git）
├── explainability/
│   ├── run_explainability.py        # 单样本分析主入口
│   ├── run_population_analysis.py   # 群体分析主入口
│   ├── llm_interpret_population.py  # 群体 LLM 解读（独立脚本）
│   ├── methods/                     # 归因方法实现
│   ├── adapters/                    # 模型适配器
│   ├── load_model/                  # 权重加载
│   ├── spectral_attribution.py      # Patch-频段相关性分析
│   ├── spectral_band_attribution.py # 频段归因
│   ├── faithfulness.py              # AOPC 忠实度评估
│   └── visualizer.py                # 可视化工具
└── requirements.txt
```

---

## 安装

```bash
# 1. 普通依赖
pip install -r requirements.txt

# 2. torch（按你的 CUDA 版本，从 pytorch.org 选命令）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 仅运行 EEGMamba 时需要（需 GPU + CUDA + 编译器，Windows 较难装）
pip install packaging ninja
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.0 --no-build-isolation
```

---

## 快速开始

### 1. 单样本分析

**方式 A：从数据集取样本**
```bash
python explainability/run_explainability.py \
    --model-type cbramod \
    --task tuab \
    --checkpoint path/to/finetuned.pth \
    --data-from-dataset path/to/dataset \
    --index 0 \
    --method gradcam \
    --output-dir ./results
```

**方式 B：直接喂单个文件**（`.npy` / `.pt` / `.mat` / `.npz`）
```bash
python explainability/run_explainability.py \
    --model-type cbramod \
    --task tuab \
    --checkpoint path/to/finetuned.pth \
    --data sample.npy \
    --method gradcam \
    --output-dir ./results
```

> `--data` 与 `--data-from-dataset` **二选一**。`.mat` / `.npz` 多键时用 `--data-key eeg` 指定键名。

**默认产出**（在 `./results/sample/...` 下）：
- 归因结果（`combined.npy`、`spatial_importance.npy`）
- 通道地形图、波形归因图
- 频段归因地形图（5 个频段）
- LLM 解读报告（`--llm` 默认 `claude`）

### 2. 群体分析

```bash
python explainability/run_population_analysis.py \
    --model-type cbramod \
    --task tuab \
    --checkpoint path/to/finetuned.pth \
    --data-from-dataset path/to/dataset \
    --method gradcam \
    --target-class 1 \
    --n-samples 50 \
    --output-dir ./population_results
```

**默认产出**：
- 群体平均通道重要度地形图
- Patch 归因 vs 频段能量 Spearman 相关性（带 t 检验显著性）
- 频段归因（每个频段的群体平均）
- 忠实度评估（AOPC 曲线）
- 群体级 JSON 汇总

### 3. 群体 LLM 解读（独立运行）

群体分析结束后，单独跑 LLM 生成报告：

```bash
python -m explainability.llm_interpret_population \
    --population-dir ./population_results \
    --llm claude
```

---

## 关键 CLI 参数

### `run_explainability.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model-type` | 必填 | `cbramod` / `labram` / `eegmamba` / `eegpt` / `biot` |
| `--task` | 必填 | `tuab` / `bciciv2a` / `stress` / ... |
| `--checkpoint` | 必填 | 微调后的模型权重 |
| `--method` | 必填 | `gradcam` / `ig` / `gradient_shap` / `shap` / `lime` / `occlusion` |
| `--data-from-dataset` | 必填 | 数据集路径（LMDB / .npy / .npz / .pt） |
| `--index` | 0 | 样本索引 |
| `--tp-class` | None | 自动找 true-positive 样本（替代 `--index`） |
| `--llm` | `claude` | LLM 后端（设 `--llm none` 禁用） |
| `--spectral-top-k` | 5 | 频段分析的 Top-K 通道数 |
| `--patch-band-corr` | off | 单样本 Patch-频段相关性分析 |
| `--no-band-topomap` | — | 禁用单样本频段地形图 |

### `run_population_analysis.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--target-class` | None | 单类分析（必填，或用 `--target-classes 0,1`） |
| `--tp-conf-threshold` | 0.7 | TP 样本置信度阈值 |
| `--fp-conf-threshold` | 0.8 | FP 样本置信度阈值 |
| `--n-samples` | -1 | 每类筛多少样本（-1 = 全部） |
| `--top-k` | 5 | Top-K 通道精细分析 |
| `--skip-band-attribution` | off | 跳过频段归因（耗时） |
| `--per-subject` | off | 按被试拆分分析 |
| `--n-workers` | 1 | 并行进程数 |

---

## 支持的归因方法

| 方法 | flag 名 | Paradigm |
|---|---|---|
| GradCAM | `gradcam` | Activation-based |
| Integrated Gradients | `ig` | Gradient-based |
| Gradient SHAP | `gradient_shap` | Gradient-based |
| SHAP | `shap` | Perturbation-based |
| LIME | `lime` | Perturbation-based |
| Occlusion | `occlusion` | Perturbation-based |

跑多个方法：`--methods gradcam,ig,shap` 或 `--all-methods`。

---

## 配置文件

每个模型对应一个 YAML（`configs/<model>.yaml`），按 task 给出 `n_channels`、`n_patches`、`num_classes`、`patch_size`、`fs` 等。新增 task 时直接在对应 YAML 的 `CONFIGS:` 下加节即可。

---

## 注意事项

- 数据集太大不要上传 GitHub，放 `data/` 目录（已在 `.gitignore`）
- 预训练权重也不上传，放 `pretrained-models/`
- 单样本路径默认会自动调用 `llm_interpret.py`；群体路径**不会**自动调用 LLM，需手动跑 `llm_interpret_population`
- mamba-ssm 在 Windows 难装，Linux + CUDA 环境最稳

---

## 扩展：加新模型 / 新方法

### 加一个新模型（示例：`myeeg`）

需要 5 个动作，全部走注册器模式：

1. **模型定义** — `model_list/myeeg.py`
   定义 `MyEEGClassifier`（`forward` 接受 `(B, C, T)` 或 `(B, C, N, patch_size)`，输出 logits）。

2. **配置文件** — `configs/myeeg.yaml`
   ```yaml
   CONFIGS:
     tuab:
       n_channels: 16
       n_patches: 30
       num_classes: 1
       patch_size: 200
       fs: 200
   ```

3. **权重加载** — `explainability/load_model/load_finetune_myeeg.py`
   写一个 `MyEEGLoader.load(checkpoint_path, ...)`，返回加载好权重的模型实例。

4. **适配器** — `explainability/adapters/myeeg_adapter.py`
   ```python
   from ..model_adapter import ModelAdapter, ModelAdapterRegistry

   @ModelAdapterRegistry.register('myeeg')
   class MyEEGAdapter(ModelAdapter):
       def get_target_layer(self): ...    # GradCAM 用
       def get_n_channels(self): ...
       def get_n_patches(self): ...
       def prepare_input(self, x): ...    # 训练时若做了归一化，这里要复现
   ```
   然后在 `explainability/adapters/__init__.py` 加 `from .myeeg_adapter import MyEEGAdapter`。

5. **模型工厂** — 在 `explainability/run_explainability.py` 注册创建函数：
   ```python
   @ModelFactory.register('myeeg')
   def create_myeeg_model(config, checkpoint_path=None, **kwargs):
       from load_model.load_finetune_myeeg import MyEEGLoader
       return MyEEGLoader.load(checkpoint_path=checkpoint_path, **config)
   ```

跑：`python explainability/run_explainability.py --model-type myeeg ...`

### 加一个新归因方法（示例：`mygrad`）

1. **方法实现** — `explainability/methods/mygrad.py`
   ```python
   from ..base import ExplainabilityMethod, ExplainabilityRegistry

   @ExplainabilityRegistry.register('mygrad')
   class MyGrad(ExplainabilityMethod):
       def explain(self, input_tensor, target=None):
           # 计算归因，返回 (B, C, T) 或 (B, C, N) 的 numpy
           ...
           return attribution
   ```

2. 在 `explainability/methods/__init__.py` 加 `from .mygrad import MyGrad`。

跑：`python explainability/run_explainability.py --method mygrad ...`

---

## License

MIT
