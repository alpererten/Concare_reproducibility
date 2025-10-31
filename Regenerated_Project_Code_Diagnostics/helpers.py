import os
import json
import random
from typing import Dict, Any, Optional

import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # Strong determinism setting in PyTorch 2.x
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"Warning: could not enable deterministic algorithms: {e}")

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
