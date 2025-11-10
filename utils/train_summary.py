from typing import Any, Dict, List, Optional
import os
import csv

import pytorch_lightning as pl


class EpochSummaryCSV(pl.Callback):
    def __init__(self):
        super().__init__()
        self.filepath: Optional[str] = None
        self.header_written: bool = False

    def _resolve_log_dir(self, trainer: pl.Trainer) -> str:
        try:
            log_dir = trainer.logger.log_dir
        except Exception:
            try:
                name = getattr(trainer.logger, "name", "GenIDP")
                version = getattr(trainer.logger, "version", "version_0")
                save_dir = getattr(trainer.logger, "save_dir", os.path.abspath("./lightning_logs"))
                version_dir = f"version_{version}" if isinstance(version, int) else str(version)
                log_dir = os.path.join(save_dir, name, version_dir)
            except Exception:
                log_dir = os.path.abspath("./lightning_logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        log_dir = self._resolve_log_dir(trainer)
        self.filepath = os.path.join(log_dir, "epoch_summary.csv")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        header: List[str] = [
            "epoch", "lr-AdamW",
            "train_loss_epoch", "train_acc", "train_hit@5", "train_hit@10", "train_hit@20",
            "train_map@5", "train_map@10", "train_map@20",
            "val_loss", "val_acc", "val_hit@5", "val_hit@10", "val_hit@20",
            "val_map@5", "val_map@10", "val_map@20",
        ]
        row: Dict[str, Any] = {k: "" for k in header}
        row["epoch"] = int(trainer.current_epoch)
        lr_key = next((k for k in metrics.keys() if k.startswith("lr-")), None)
        if lr_key is not None:
            try:
                row["lr-AdamW"] = float(metrics[lr_key])
            except Exception:
                row["lr-AdamW"] = metrics[lr_key]
        for k in header:
            if k in metrics:
                try:
                    row[k] = float(metrics[k])
                except Exception:
                    row[k] = metrics[k]
        write_header = not self.header_written or not os.path.exists(self.filepath)
        with open(self.filepath, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
                self.header_written = True
            w.writerow([row[k] for k in header])


def write_final_report(trainer: pl.Trainer, args_model_name: str, val_out, test_out) -> str:
    def _resolve_log_dir() -> str:
        try:
            log_dir = trainer.logger.log_dir
        except Exception:
            try:
                name = getattr(trainer.logger, "name", args_model_name)
                version = getattr(trainer.logger, "version", "version_0")
                save_dir = getattr(trainer.logger, "save_dir", os.path.abspath("./lightning_logs"))
                version_dir = f"version_{version}" if isinstance(version, int) else str(version)
                log_dir = os.path.join(save_dir, name, version_dir)
            except Exception:
                log_dir = os.path.abspath("./lightning_logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _fmt(v):
        try:
            return f"{float(v):.6f}"
        except Exception:
            return str(v)

    def _pick(d: Dict[str, Any], keys):
        out = {}
        for k in keys:
            if k in d:
                out[k] = d[k]
        return out

    log_dir = _resolve_log_dir()
    val_dict = val_out[0] if isinstance(val_out, (list, tuple)) and len(val_out) > 0 and isinstance(val_out[0], dict) else {}
    test_dict = test_out[0] if isinstance(test_out, (list, tuple)) and len(test_out) > 0 and isinstance(test_out[0], dict) else {}

    final_path = os.path.join(log_dir, "final_report.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(f"=== Training/Validation Summary ({args_model_name}) ===\n")
        val_keys = [
            "val_loss", "val_acc",
            "val_hit@5", "val_hit@10", "val_hit@20",
            "val_map@5", "val_map@10", "val_map@20",
        ]
        vals = _pick(val_dict, val_keys)
        if vals:
            f.write("Validation (best ckpt):\n")
            f.write("  " + ", ".join([f"{k}={_fmt(v)}" for k, v in vals.items()]) + "\n")
        else:
            f.write("Validation (best ckpt): (no metrics)\n")

        f.write("\n=== Final Test Summary ===\n")
        test_keys = [
            "test_loss", "test_acc",
            "test_hit@5", "test_hit@10", "test_hit@20",
            "test_map@5", "test_map@10", "test_map@20",
        ]
        tests = _pick(test_dict, test_keys)
        if tests:
            f.write("  " + ", ".join([f"{k}={_fmt(v)}" for k, v in tests.items()]) + "\n")
        else:
            f.write("  (no test metrics)\n")

    try:
        print(open(final_path, "r", encoding="utf-8").read())
    except Exception:
        pass

    return final_path
    