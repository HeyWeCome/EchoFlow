#!/usr/bin/env python3
import argparse
import os
import warnings
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping

from pl.data_module import CascadesDataModule
from utils.diffusion_events import build_training_events_from_dm
import yaml
import torch
import importlib
from utils.train_summary import EpochSummaryCSV, write_final_report


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _override(base: Dict[str, Any], cli: Dict[str, Any]) -> Dict[str, Any]:
    out = {**base}
    for k, v in cli.items():
        if v is None:
            continue
        out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train models for cascade next-user prediction")
    group_data = p.add_argument_group("data")
    group_data.add_argument("--dataset", choices=["zhihu", "douban", "yelp", "movie", "android", "christianity", "twitter", "memetracker", "pheme_9", "rumoreval_2019_reddit", "twitter15", "twitter16"], help="Dataset shortcut name")
    group_data.add_argument("--interactions_path", help="Explicit path to interactions.inter (overrides --dataset)")
    group_data.add_argument("--root", default="dataset", help="Root dir for --dataset shortcut (default: dataset)")
    group_data.add_argument("--min_len", type=int, default=4, help="Minimum cascade length (default: 4)")
    group_data.add_argument("--batch_size", type=int, default=128, help="Batch size")
    group_data.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    group_data.add_argument("--pin_memory", action="store_true", default=None, help="Enable DataLoader pin_memory")
    group_data.add_argument("--persistent_workers", action="store_true", default=None, help="Enable persistent workers (requires num_workers>0)")
    group_data.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch_factor (requires num_workers>0)")
    group_data.add_argument("--train_style", choices=["seq", "pairs"], default="pairs", help="Training style: sequence loss or pairwise target with negative sampling")
    group_data.add_argument("--window_enabled", action="store_true", default=None, help="Enable sliding window sampling for long cascades")
    group_data.add_argument("--window_len_mode", choices=["auto", "fixed"], default=None, help="Window length mode")
    group_data.add_argument("--window_len_cap", type=int, default=None, help="Max window length cap")
    group_data.add_argument("--window_stride", type=int, default=None, help="Window stride; if unset, use ratio")
    group_data.add_argument("--window_stride_ratio", type=float, default=None, help="Window stride ratio when stride is unset")

    group_model = p.add_argument_group("model")
    group_model.add_argument("--model_name", type=str, default="baseline", help="Model name (located at pl/models/<name>.py, must provide build_model(...))")
    group_model.add_argument("--d_model", type=int, default=64)
    group_model.add_argument("--n_heads", type=int, default=4)
    group_model.add_argument("--n_layers", type=int, default=2)
    group_model.add_argument("--dropout", type=float, default=0.1)
    group_model.add_argument("--topk", type=int, nargs="+", default=None, help="Top-k list for metrics (e.g., 5 10 20)")
    group_model.add_argument("--diffusion_steps", type=int, default=None, help="SeqDiff: number of diffusion steps (e.g., 5; None disables diffusion)")
    group_model.add_argument("--diffusion_schedule", choices=["linear", "cosine"], default=None, help="SeqDiff: beta schedule type")
    group_model.add_argument("--condition_xt", action="store_true", default=None, help="SeqDiff: condition on corrupted label x_t")
    group_model.add_argument("--mc_samples", type=int, default=None, help="SeqDiff: MC Dropout samples at eval (0 to disable)")
    group_model.add_argument("--temperature", type=float, default=None, help="SeqDiff: inference temperature scaling")

    group_opt = p.add_argument_group("optim")
    group_opt.add_argument("--lr", type=float, default=1e-3)
    group_opt.add_argument("--weight_decay", type=float, default=0.01)

    group_trainer = p.add_argument_group("trainer")
    group_trainer.add_argument("--max_epochs", type=int, default=300)
    group_trainer.add_argument("--seed", type=int, default=42)
    group_trainer.add_argument("--default_root_dir", default=os.path.abspath("./lightning_logs"))
    group_trainer.add_argument("--accelerator", default=None)
    group_trainer.add_argument("--devices", default=None)
    group_trainer.add_argument("--log_every_n_steps", type=int, default=None)
    group_trainer.add_argument("--precision", type=str, default=None, help="Precision for Trainer (e.g., 32, 16, bf16)")

    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")

    return p.parse_args()


def main():
    args = parse_args()
    # Reduce PyTorch Lightning warnings and other noisy logs
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="pytorch_lightning")

    # Load YAML and apply CLI overrides (CLI has higher priority)
    cfg = _load_yaml(args.config)
    data_cfg = _override(cfg.get("data", {}), {
        "dataset": args.dataset,
        "interactions_path": args.interactions_path,
        "root": args.root,
        "min_len": args.min_len,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "train_style": args.train_style,
        "train_max_len": None,
        "eval_max_len": None,
        "window_enabled": args.window_enabled,
        "window_len_mode": args.window_len_mode,
        "window_len_cap": args.window_len_cap,
        "window_stride": args.window_stride,
        "window_stride_ratio": args.window_stride_ratio,
    })
    model_cfg = _override(cfg.get("model", {}), {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "topk": args.topk,
        "diffusion_steps": args.diffusion_steps,
        "diffusion_schedule": args.diffusion_schedule,
        "condition_xt": args.condition_xt,
        "mc_samples": args.mc_samples,
        "temperature": args.temperature,
    })
    optim_cfg = _override(cfg.get("optim", {}), {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    })
    trainer_cfg = _override(cfg.get("trainer", {}), {
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "default_root_dir": args.default_root_dir,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "log_every_n_steps": args.log_every_n_steps,
        "precision": args.precision,
    })

    pl.seed_everything(trainer_cfg.get("seed", 42), workers=True)

    dm = CascadesDataModule(
        interactions_path=data_cfg.get("interactions_path"),
        dataset=data_cfg.get("dataset"),
        root=data_cfg.get("root", "dataset"),
        min_len=int(data_cfg.get("min_len", 4)),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        train_style=str(data_cfg.get("train_style", "seq")),
        train_max_len=data_cfg.get("train_max_len"),
        eval_max_len=data_cfg.get("eval_max_len"),
        window_enabled=bool(data_cfg.get("window_enabled", False)),
        window_len_mode=str(data_cfg.get("window_len_mode", "auto")),
        window_len_cap=data_cfg.get("window_len_cap", 64),
        window_stride=data_cfg.get("window_stride"),
        window_stride_ratio=data_cfg.get("window_stride_ratio", 0.5),
    )

    # Prepare to know vocab size
    dm.setup("fit")
    vocab_size = dm.num_users
    # When using time-graph models, iterate samples in chronological order to avoid leakage (event pointer only moves forward)
    # Enabled only for the tgn_seq model; other models keep their original behavior
    if args.model_name == "tgn_seq":
        dm.time_order = True

    # Dynamic model loading: import pl.models.<name>.py based on --model_name and call its build_model
    def _build_model_dynamic(name: str):
        mod_path = f"pl.models.{name}"
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            # Try developing fallback: pl.models.developing.<name>
            dev_path = f"pl.models.developing.{name}"
            mod = importlib.import_module(dev_path)
        builder = None
        if hasattr(mod, "build_model"):
            builder = getattr(mod, "build_model")
        elif hasattr(mod, "build_baseline_model"):
            builder = getattr(mod, "build_baseline_model")
        else:
            raise AttributeError(f"Model module {mod_path} does not provide build_model/build_baseline_model factory functions")
        if name == "dyhgcn":
            events = build_training_events_from_dm(dm)
            return builder(
                vocab_size=vocab_size,
                model_cfg=model_cfg,
                optim_cfg=optim_cfg,
                events=events,
            )
        elif name == "mshgat":
            events = build_training_events_from_dm(dm)
            return builder(
                vocab_size=vocab_size,
                model_cfg=model_cfg,
                optim_cfg=optim_cfg,
                events=events,
            )
        elif name == "sieve":
            events = build_training_events_from_dm(dm)
            return builder(
                vocab_size=vocab_size,
                model_cfg=model_cfg,
                optim_cfg=optim_cfg,
                events=events,
                num_topics=dm.num_topics,
            )
        elif name == "pmrca":
            events = build_training_events_from_dm(dm)
            return builder(
                vocab_size=vocab_size,
                model_cfg=model_cfg,
                optim_cfg=optim_cfg,
                events=events,
                num_topics=dm.num_topics,
            )
        else:
            return builder(
                vocab_size=vocab_size,
                model_cfg=model_cfg,
                optim_cfg=optim_cfg,
            )

    model = _build_model_dynamic(args.model_name)
    # topk has been written into hparams in builder

    ckpt_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="epoch{epoch}-val_acc{val_acc:.4f}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    es_cfg = trainer_cfg.get("early_stopping", {})
    es_cb = EarlyStopping(
        monitor=es_cfg.get("monitor", "val_acc"),
        mode=es_cfg.get("mode", "max"),
        patience=int(es_cfg.get("patience", 10)),
    )

    # Use tidy CSV callback from utils
    summary_cb = EpochSummaryCSV()

    # Optional acceleration: mixed precision and TF32 for matmul on Ampere+
    # Respect YAML settings if provided; fall back to sensible defaults
    prec_cfg = trainer_cfg.get("precision", 32)
    precision = prec_cfg if isinstance(prec_cfg, str) else int(prec_cfg)
    # Enable TF32 when using float32 matmul to accelerate training on modern GPUs
    if trainer_cfg.get("enable_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Prepare Trainer
    logger_type = trainer_cfg.get("logger", "tensorboard")
    logger_name = args.model_name
    logger_version = trainer_cfg.get("logger_version", None)
    default_root = os.path.abspath(trainer_cfg.get("default_root_dir", "./lightning_logs"))
    if logger_type == "tensorboard":
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            tb_logger = TensorBoardLogger(save_dir=default_root, name=logger_name, version=logger_version)
            logger_obj = tb_logger
        except Exception:
            logger_obj = None
    elif logger_type == "csv":
        try:
            from pytorch_lightning.loggers import CSVLogger
            csv_logger = CSVLogger(save_dir=default_root, name=logger_name, version=logger_version)
            logger_obj = csv_logger
        except Exception:
            logger_obj = None
    else:
        logger_obj = None

    trainer = Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 5)),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        default_root_dir=default_root,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        precision=precision,
        callbacks=[ckpt_cb, lr_cb, es_cb, summary_cb],
        logger=logger_obj,
    )

    trainer.fit(model, datamodule=dm)
    val_out = trainer.validate(model, datamodule=dm, ckpt_path="best") or []
    test_out = trainer.test(model, datamodule=dm, ckpt_path="best") or []

    # Use final report generator from utils
    final_path = write_final_report(trainer, args.model_name, val_out, test_out)


if __name__ == "__main__":
    main()
