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
    
    # Allow overriding any config via key-value pairs (e.g. --data.batch_size 64)
    # This is a more flexible approach than defining every single argument
    p.add_argument("overrides", nargs="*", help="Override config values (e.g., data.batch_size=64)")

    args = p.parse_args()
    
    # Load base YAML config
    cfg = _load_yaml(args.config)
    
    # Apply CLI model_name if provided (it's a special top-level arg)
    if args.model_name:
        if "model" not in cfg:
            cfg["model"] = {}
        # We store model_name in the config root for easy access, 
        # but also ensure it's available where needed
        cfg["model_name"] = args.model_name

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
            
        # Nested set
        keys = key.split(".")
        current = cfg
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return cfg
