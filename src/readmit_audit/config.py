"""Load and validate the pipeline configuration.

All relative paths in the config (data, output directories) are resolved against the folder that holds
config.yaml, never against the current working directory. That way scripts work from any folder after a
fresh `git clone`.
"""
from pathlib import Path
from typing import Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

_PATH_KEYS = (("data", "path"), ("output", "results_dir"), ("output", "figures_dir"))


def _find_config(path: Optional[str]) -> Path:
    if path is None:
        return DEFAULT_CONFIG
    p = Path(path)
    if p.exists():
        return p.resolve()
    alt = PROJECT_ROOT / path          # a relative name like "config.yaml" given from another folder
    return alt if alt.exists() else p


def load_config(path: Optional[str] = None) -> dict:
    """Load the YAML config and resolve its relative paths against the config's own folder.

    Raises FileNotFoundError with a clear message if the config or the data file is missing.
    """
    p = _find_config(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path or DEFAULT_CONFIG}'. Pass an explicit --config path."
        )
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    root = p.parent
    cfg["_root"] = str(root)
    for section, key in _PATH_KEYS:
        value = cfg.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(root / value)
    data_path = Path(cfg["data"]["path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{data_path}'. Expected data/diabetic_data.csv inside the project folder; "
            f"download it from https://archive.ics.uci.edu/dataset/296 if it is missing."
        )
    return cfg
