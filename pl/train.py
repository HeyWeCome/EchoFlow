#!/usr/bin/env python3
import os
import warnings
import importlib
import inspect
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping

from pl.data_module import CascadesDataModule
from utils.diffusion_events import build_training_events_from_dm
from utils.train_summary import EpochSummaryCSV, write_final_report
from utils.config import get_config


def main():
    # Reduce PyTorch Lightning warnings and other noisy logs
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="pytorch_lightning")

    # Load unified configuration (YAML + CLI overrides)
    cfg = get_config()
    
    # Extract config sections
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optim", {})
    trainer_cfg = cfg.get("trainer", {})
    
    # Global settings
    model_name = cfg.get("model_name", model_cfg.get("model_name", "baseline"))
    
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
    if model_name == "tgn_seq":
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

        # Prepare arguments dynamically based on builder signature
        sig = inspect.signature(builder)
        kwargs = {
            "vocab_size": vocab_size,
            "model_cfg": model_cfg,
            "optim_cfg": optim_cfg,
        }
        
        # Inject optional dependencies if requested by the builder
        if "events" in sig.parameters:
            kwargs["events"] = build_training_events_from_dm(dm)
        if "num_topics" in sig.parameters:
            kwargs["num_topics"] = dm.num_topics
            
        return builder(**kwargs)

    model = _build_model_dynamic(model_name)
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
    logger_name_val = model_name
    logger_version = trainer_cfg.get("logger_version", None)
    default_root = os.path.abspath(trainer_cfg.get("default_root_dir", "./lightning_logs"))
    if logger_type == "tensorboard":
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            tb_logger = TensorBoardLogger(save_dir=default_root, name=logger_name_val, version=logger_version)
            logger_obj = tb_logger
        except Exception:
            logger_obj = None
    elif logger_type == "csv":
        try:
            from pytorch_lightning.loggers import CSVLogger
            csv_logger = CSVLogger(save_dir=default_root, name=logger_name_val, version=logger_version)
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
    final_path = write_final_report(trainer, model_name, val_out, test_out)


if __name__ == "__main__":
    main()
