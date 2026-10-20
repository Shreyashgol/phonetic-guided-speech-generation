import os
import yaml


def load_config(project_root, name="data.yaml"):
    """Load a YAML config from <project_root>/configs and add absolute paths."""
    with open(os.path.join(project_root, "configs", name)) as f:
        cfg = yaml.safe_load(f)
    cfg["abs_paths"] = {
        k: os.path.join(project_root, v) for k, v in cfg["paths"].items()
    }
    return cfg
