import argparse
import os
import yaml
from typing import Any, Dict, Optional

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

def get_config() -> Dict[str, Any]:
    """
    Parses CLI arguments and merges them with YAML configuration.
    Returns a unified configuration dictionary.
    """
    p = argparse.ArgumentParser(description="Train models for cascade next-user prediction")
    
    # Only minimal required args in CLI
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    p.add_argument("--model_name", type=str, default="baseline", help="Model name")
    p.add_argument("--dataset", type=str, help="Dataset name (alias for data.dataset)")
    
    # Allow overriding any config via key-value pairs (e.g. --data.batch_size 64)
    # Also support simplified args like --batch_size 64 if unique
    p.add_argument("overrides", nargs="*", help="Override config values (e.g., batch_size=64)")

    args = p.parse_args()
    
    # Load base YAML config
    cfg = _load_yaml(args.config)
    
    # Apply CLI model_name if provided (it's a special top-level arg)
    if args.model_name:
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model_name"] = args.model_name

    # Apply dataset alias if provided
    if args.dataset:
        if "data" not in cfg:
            cfg["data"] = {}
        cfg["data"]["dataset"] = args.dataset

    # Process generic overrides
    for override in args.overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        
        # Try to parse value as int/float/bool/list
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "none":
                value = None
            elif "," in value and "[" in value and "]" in value:
                 # simple list parsing like [1,2,3]
                 import ast
                 value = ast.literal_eval(value)
            elif "." in value:
                value = float(value)
            else:
                value = int(value)
        except (ValueError, SyntaxError):
            pass  # keep as string
            
        # Support simplified keys (e.g. batch_size -> data.batch_size)
        # We search for the key in the default config structure
        full_key = key
        if "." not in key:
            # Common shortcuts map
            shortcuts = {
                "batch_size": "data.batch_size",
                "lr": "optim.lr",
                "epochs": "trainer.max_epochs",
                "gpu": "trainer.accelerator", 
                "seed": "trainer.seed",
                "dropout": "model.dropout",
                "layers": "model.n_layers",
                "heads": "model.n_heads",
                "dim": "model.d_model",
            }
            if key in shortcuts:
                full_key = shortcuts[key]
            else:
                # Try to infer from existing config structure
                # This is a simple BFS/DFS to find a unique match
                found_paths = []
                def search_key(d, target, prefix=""):
                    for k, v in d.items():
                        path = f"{prefix}.{k}" if prefix else k
                        if k == target:
                            found_paths.append(path)
                        if isinstance(v, dict):
                            search_key(v, target, path)
                
                search_key(cfg, key)
                if len(found_paths) == 1:
                    full_key = found_paths[0]
                # If ambiguous or not found, we treat it as a new top-level or sub-level key as requested
                # But for safety, if it's not found, we just use it as is (maybe user added new param)

        # Nested set
        keys = full_key.split(".")
        current = cfg
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return cfg
