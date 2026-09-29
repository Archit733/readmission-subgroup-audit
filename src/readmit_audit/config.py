"""Load and validate the pipeline configuration."""
from pathlib import Path
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load the YAML config. Raises FileNotFoundError with a clear message if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. Run scripts from the project root, "
            f"or pass an explicit --config path."
        )
    with open(p, "r") as f:
        return yaml.safe_load(f)
