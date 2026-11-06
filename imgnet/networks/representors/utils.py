import os
from pathlib import Path

MODEL_ZOO = Path(os.environ.get("DM_MODEL_ZOO", "/path/to/your/model_zoo"))


def pretrained_path(*parts):
    return str(MODEL_ZOO / "pretrained" / Path(*parts))
