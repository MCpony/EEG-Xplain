# EEG-Xplain

An explainability analysis framework for EEG foundation models. Supports multiple EEG models, attribution methods, single-sample and population-level analysis, and LLM-based interpretation.

---

## Features

- **Multi-model support**: CBraMod, LaBraM, EEGMamba, EEGPT, BIOT
- **Multi-method attribution**: GradCAM, Integrated Gradients, Gradient SHAP, SHAP, LIME, Occlusion
- **Multi-dimensional analysis**
  - Single-sample attribution (channel topomap, waveform attribution)
  - Population-level attribution (cross-sample average + statistical testing)
  - Spectral band attribution (Delta / Theta / Alpha / Beta / Gamma)
  - Patch attribution vs. band energy correlation (Spearman ρ)
  - Faithfulness evaluation (AOPC)
- **LLM interpretation**: Auto-generate readable reports via Claude / OpenAI / DeepSeek

---

## Project Structure

```
EEG-Xplain/
├── configs/                  # Model YAML configs
├── model_list/               # Model definitions
├── pretrained-models/        # Pretrained weights (not tracked by git)
├── data/                     # Data (not tracked by git)
├── explainability/
│   ├── run_explainability.py        # Single-sample analysis entry point
│   ├── run_population_analysis.py   # Population analysis entry point
│   ├── llm_interpret_population.py  # Population LLM interpretation (standalone)
│   ├── methods/                     # Attribution method implementations
│   ├── adapters/                    # Model adapters
│   ├── load_model/                  # Checkpoint loaders
│   ├── spectral_attribution.py      # Patch–band correlation analysis
│   ├── spectral_band_attribution.py # Spectral band attribution
│   ├── faithfulness.py              # AOPC faithfulness evaluation
│   └── visualizer.py                # Visualization utilities
└── requirements.txt
```

---

## Installation

```bash
# 1. Standard dependencies
pip install -r requirements.txt

# 2. PyTorch — select the command matching your CUDA version at pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. EEGMamba only — requires GPU + CUDA toolkit + compiler (difficult on Windows)
pip install packaging ninja
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.0 --no-build-isolation
```

---

## Quick Start

### 1. Single-sample Analysis

**Option A: Load from dataset**
```bash
python explainability/run_explainability.py \
    --model-type cbramod \
    --task tuab \
    --checkpoint path/to/finetuned.pth \
    --data-from-dataset path/to/dataset \
    --index 0 \
    --method ig \
    --output-dir ./results \
    --api-key your_api_key \
    --api-base relay_url  # omit to use the official endpoint
```

**Option B: Single file** (`.npy` / `.pt` / `.mat` / `.npz`)
```bash
python explainability/run_explainability.py \
    --model-type cbramod \
    --task tuab \
    --checkpoint path/to/finetuned.pth \
    --data sample.npy \
    --method ig \
    --output-dir ./results \
    --api-key your_api_key \
    --api-base relay_url  # omit to use the official endpoint
```

> `--data` and `--data-from-dataset` are mutually exclusive. For `.mat` / `.npz` files with multiple keys, specify the key with `--data-key eeg`.

**Default outputs** (under `./results/sample/...`):
- Attribution arrays (`combined.npy`, `spatial_importance.npy`)
- Channel topomap and waveform attribution plots
- Spectral band topomap (5 bands)
- LLM interpretation report (`--llm` defaults to `claude`)

### 2. Population Analysis

```bash
python explainability/run_population_analysis.py \
    --model-type cbramod \
    --task tuab \
    --checkpoint path/to/finetuned.pth \
    --data-from-dataset path/to/dataset \
    --method ig \
    --band-methods ig \
    --band-baseline zero \
    --target-class 1 \
    --n-samples 100 \
    --output-dir ./population_results
```

**Default outputs**:
- Population-average channel importance topomap
- Patch attribution vs. band energy Spearman correlation (with t-test significance)
- Band attribution (population average per band)
- Faithfulness evaluation (AOPC curve)
- Population-level JSON summary

### 3. Population LLM Interpretation (standalone)

Run separately after population analysis:

```bash
python -m explainability.llm_interpret_population \
    --population-dir ./population_results \
    --llm claude \
    --api-key your_api_key \
    --api-base relay_url  # omit to use the official endpoint
```

---

## Key CLI Arguments

### `run_explainability.py`

| Argument | Default | Description |
|---|---|---|
| `--model-type` | required | `cbramod` / `labram` / `eegmamba` / `eegpt` / `biot` |
| `--task` | required | `tuab` / `bciciv2a` / `stress` / ... |
| `--checkpoint` | required | Path to fine-tuned model weights |
| `--method` | required | `gradcam` / `ig` / `gradient_shap` / `shap` / `lime` / `occlusion` |
| `--data-from-dataset` | — | Dataset path (LMDB / .npy / .npz / .pt) |
| `--data` | — | Single sample file (.npy / .pt / .mat / .npz) |
| `--index` | 0 | Sample index when using `--data-from-dataset` |
| `--tp-class` | None | Auto-find a true-positive sample (replaces `--index`) |
| `--llm` | `claude` | LLM backend (`--llm none` to disable) |
| `--spectral-top-k` | 5 | Top-K channels for band / patch-band analysis |
| `--patch-band-corr` | off | Run single-sample patch–band correlation |
| `--no-band-topomap` | — | Disable single-sample band topomap |

### `run_population_analysis.py`

| Argument | Default | Description |
|---|---|---|
| `--target-class` | required | Target class index (or use `--target-classes 0,1`) |
| `--tp-conf-threshold` | 0.7 | Confidence threshold for TP samples |
| `--fp-conf-threshold` | 0.8 | Confidence threshold for FP samples |
| `--n-samples` | -1 | Samples per class (-1 = all) |
| `--top-k` | 5 | Top-K channels for detailed analysis |
| `--band-methods` | None | Attribution method(s) for band analysis |
| `--band-baseline` | `auto` | Baseline strategy for band attribution (`zero` / `auto` / `class_permute`) |
| `--skip-band-attribution` | off | Skip band attribution (saves time) |
| `--per-subject` | off | Run per-subject breakdown |
| `--n-workers` | 1 | Number of parallel workers |

---

## Supported Attribution Methods

| Method | Flag | Paradigm |
|---|---|---|
| GradCAM | `gradcam` | Activation-based |
| Integrated Gradients | `ig` | Gradient-based |
| Gradient SHAP | `gradient_shap` | Gradient-based |
| SHAP | `shap` | Perturbation-based |
| LIME | `lime` | Perturbation-based |
| Occlusion | `occlusion` | Perturbation-based |

Run multiple methods: `--methods gradcam,ig,shap` or `--all-methods`.

---

## Configuration Files

Each model has a YAML file (`configs/<model>.yaml`) with per-task settings: `n_channels`, `n_patches`, `num_classes`, `patch_size`, `fs`, etc. To add a new task, add an entry under `CONFIGS:` in the corresponding YAML.

---

## Notes

- Single-sample analysis calls `llm_interpret.py` automatically (when `--llm` is set). Population analysis does **not** auto-call LLM — run `llm_interpret_population` manually.
- `mamba-ssm` is difficult to install on Windows; a Linux + CUDA environment is recommended.
- Do not commit large data files or model weights to git. Keep them in `data/` and `pretrained-models/` (both in `.gitignore`).

---

## Extension: Adding New Models / Methods

### Adding a new model (example: `myeeg`)

Five steps, all using the registry pattern:

1. **Model definition** — `model_list/myeeg.py`
   Define `MyEEGClassifier` (`forward` takes `(B, C, T)` or `(B, C, N, patch_size)`, returns logits).

2. **Config file** — `configs/myeeg.yaml`
   ```yaml
   CONFIGS:
     tuab:
       n_channels: 16
       n_patches: 30
       num_classes: 1
       patch_size: 200
       fs: 200
   ```

3. **Checkpoint loader** — `explainability/load_model/load_finetune_myeeg.py`
   Implement `MyEEGLoader.load(checkpoint_path, ...)` returning a model with loaded weights.

4. **Adapter** — `explainability/adapters/myeeg_adapter.py`
   ```python
   from ..model_adapter import ModelAdapter, ModelAdapterRegistry

   @ModelAdapterRegistry.register('myeeg')
   class MyEEGAdapter(ModelAdapter):
       def get_target_layer(self): ...    # required for GradCAM
       def get_n_channels(self): ...
       def get_n_patches(self): ...
       def prepare_input(self, x): ...    # reproduce any training-time normalization
   ```
   Then add `from .myeeg_adapter import MyEEGAdapter` to `explainability/adapters/__init__.py`.

5. **Model factory** — register in `explainability/run_explainability.py`:
   ```python
   @ModelFactory.register('myeeg')
   def create_myeeg_model(config, checkpoint_path=None, **kwargs):
       from load_model.load_finetune_myeeg import MyEEGLoader
       return MyEEGLoader.load(checkpoint_path=checkpoint_path, **config)
   ```

Usage: `python explainability/run_explainability.py --model-type myeeg ...`

### Adding a new attribution method (example: `mygrad`)

1. **Method implementation** — `explainability/methods/mygrad.py`
   ```python
   from ..base import ExplainabilityMethod, ExplainabilityRegistry

   @ExplainabilityRegistry.register('mygrad')
   class MyGrad(ExplainabilityMethod):
       def explain(self, input_tensor, target=None):
           # compute attribution, return numpy array (B, C, T) or (B, C, N)
           ...
           return attribution
   ```

2. Add `from .mygrad import MyGrad` to `explainability/methods/__init__.py`.

Usage: `python explainability/run_explainability.py --method mygrad ...`

---

## License

MIT
