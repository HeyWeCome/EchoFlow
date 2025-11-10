# EchoFlow

A modular, research-friendly toolkit for microscopic information diffusion prediction — focused on next-user forecasting inside information cascades. EchoFlow brings together strong sequence baselines and time-aware graph models, consistent evaluation, and simple experiment management with PyTorch Lightning.

Highlights
- Model zoo: `cyan_rnn`, `dyhgcn`, `mshgat`, `sieve`
- Time-aware training: automatic time-bin selection, diffusion event builders
- Clean data interface: CSV cascades with `user_id, topic_id, timestamp`
- Repro-friendly: unified metrics (Hit@k, MAP), CSV summaries, final report files
- Easy to extend: drop-in model factories via `pl/models/<name>.py`

Table of Contents
- Overview
- Quick Start
- Data Format
- Configuration
- Training & Evaluation
- Models
- Extending
- Project Structure
- Contributing
- License

## Overview
EchoFlow targets the microscopic information diffusion task: given a cascade (topic and user sequence over time), predict the next user. It includes:
- Sequence baselines and graph-temporal models
- Unified trainer built on PyTorch Lightning
- Consistent evaluation (Hit@k, MAP) and logging
- Ready-to-use sample datasets

## Quick Start
Prerequisites
- Python 3.8+
- PyTorch (recommended 2.0+)
- PyTorch Lightning (recommended 2.x)

Install
```bash
pip install -U pip setuptools wheel
pip install torch pytorch-lightning pyyaml
# Optional: for TensorBoard logging
pip install tensorboard
```

Run a model
```bash
# Train and evaluate DyHGCN on the built-in Twitter dataset
python -m pl.train \
  --dataset twitter \
  --model_name dyhgcn \
  --config configs/default.yaml

# Alternative: run MS-HGAT
python -m pl.train --dataset android --model_name mshgat --config configs/default.yaml

# Sequence baseline (CYAN-RNN)
python -m pl.train --dataset memetracker --model_name cyan_rnn --config configs/default.yaml
```

Notes
- `--model_name` must be one of the models present in `pl/models/`: `cyan_rnn`, `dyhgcn`, `mshgat`, `sieve`.
- You can override any YAML option via CLI flags (see Configuration below).
- Logs, checkpoints, CSV summaries, and a `final_report.txt` are saved under `lightning_logs/<model>/<version>/`.

## Data Format
EchoFlow expects per-dataset directories under `dataset/` with a single file `interactions.inter` in CSV format.

Required header and columns:
```csv
user_id,topic_id,timestamp
u_001,t_078,1692201000
u_314,t_078,1692201200
u_001,t_078,1692201400
```

Built-in samples
- `dataset/twitter/interactions.inter`
- `dataset/memetracker/interactions.inter`
- `dataset/android/interactions.inter`
- `dataset/christianity/interactions.inter`

Bring your own data
- Create `dataset/<your_dataset>/interactions.inter` with the CSV header above.
- Use `--dataset <your_dataset>` or set `--interactions_path` to a file path directly.

## Configuration
Experiments are configured via YAML and can be overridden by CLI flags.

Default config (`configs/default.yaml`):
```yaml
data:
  dataset: douban
  interactions_path: null
  root: dataset
  min_len: 4
  batch_size: 512
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  train_max_len: 20
  eval_max_len: 20

model:
  d_model: 64
  n_heads: 4
  n_layers: 2
  dropout: 0.1
  topk: [5, 10, 20]
  mask_prob: 0.25
  refine_steps: 5
  # Time-step auto partitioning: adaptively choose K by timestamp distribution
  time_step_split: auto
  auto_min_bins: 3
  auto_max_bins: 10

optim:
  lr: 0.001
  weight_decay: 0.01

trainer:
  max_epochs: 300
  seed: 42
  default_root_dir: ./lightning_logs
  accelerator: auto
  devices: auto
  log_every_n_steps: 50
  precision: 32
  enable_tf32: true
  early_stopping:
    monitor: val_acc
    mode: max
    patience: 10
  # Run logs: use name/version to differentiate experiment directories
  logger: csv
  logger_name: GenIDP
  # Optional fixed version; leave null to auto-increment as version_0/1/..
  logger_version: null
```

CLI overrides (examples):
```bash
# Change batch size and precision
python -m pl.train --dataset twitter --model_name dyhgcn --batch_size 1024 --precision bf16

# Use a specific file path instead of dataset shortcut
python -m pl.train --interactions_path /abs/path/to/interactions.inter --model_name sieve

# Customize metrics top-k
python -m pl.train --dataset memetracker --model_name cyan_rnn --topk 5 10 50
```

## Training & Evaluation
Trainer
- Uses PyTorch Lightning `Trainer` with `ModelCheckpoint`, `EarlyStopping`, and `LearningRateMonitor`.
- Mixed precision and TF32 acceleration can be enabled via config (see `precision`, `enable_tf32`).

Outputs
- Checkpoints: best by `val_acc`
- CSV summary per epoch: `epoch_summary.csv`
- Final summary report: `final_report.txt` with validation and test metrics
- Logs directory: `lightning_logs/<model>/<version>/`

Metrics
- Hit@k and MAP@k (`utils.metrics.compute_hit_map_at_k`)
- Accuracy (top-1)

## Models
Supported models (set via `--model_name`):
- `cyan_rnn`: Sequence baseline (GRU/LSTM configurable via `rnn_type`).
- `dyhgcn`: Diffusion-aware GCN with time-bin snapshots and attention.
- `mshgat`: Multi-Stage Hierarchical Graph Attention over time.
- `sieve`: Topic-aware diffusion model with contrastive components.

Time-aware models (`dyhgcn`, `mshgat`, `sieve`) use training diffusion events derived from the first L-2 items of each cascade to avoid leakage.

## Extending
Add a new model by creating `pl/models/<your_name>.py` with a factory:
```python
def build_model(vocab_size: int, model_cfg: dict, optim_cfg: dict, **kwargs):
    # return a pl.LightningModule
    ...
```

Conventions
- Sequence models (e.g., `cyan_rnn`) only need `vocab_size`, `model_cfg`, `optim_cfg`.
- Time-aware models should accept `events` (list of `(user_idx, topic_idx, ts)`) and optionally `num_topics`.
- Set `hparams.topk` from `model_cfg` for consistent metric computation.
- Use `AdamW` with `lr`, `weight_decay` from `optim_cfg` for consistency.

Dynamic loading
- The trainer imports `pl.models.<name>` and calls `build_model(...)` (or `build_baseline_model(...)` if provided).
- Example builder signatures in the repo: see `cyan_rnn.py`, `dyhgcn.py`, `mshgat.py`, `sieve.py`.

## Project Structure
```
EchoFlow/
├── configs/               # YAML configs for experiments
├── dataset/               # Built-in datasets (CSV interactions.inter)
├── pl/
│   ├── data_module.py     # CascadesDataModule: reading, encoding, splits
│   ├── layers/            # Core layers: time bins, bipartite graphs, transformer PE
│   ├── models/            # Model zoo: cyan_rnn, dyhgcn, mshgat, sieve
│   └── train.py           # Training/validation/test entrypoint (CLI + YAML)
└── utils/
    ├── metrics.py         # Hit@k, MAP@k, accuracy
    ├── diffusion_events.py# Build training diffusion events from cascades
    └── train_summary.py   # CSV epoch summaries and final report writer
```

## Contributing
Welcome! Please:
- Use clear commit messages (e.g., `feat(model): add new attention block`).
- Keep style consistent and add docstrings/comments where helpful.
- Provide small, reproducible datasets or script snippets when reporting issues.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

—

If you use EchoFlow in academic work or production, please consider starring the repo. Contributions and feedback are warmly appreciated.
