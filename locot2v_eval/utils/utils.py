import json
import re
import numpy as np
import shutil
from pathlib import Path
from typing import List


def normalize_negative_exponential(orig_val, scale_k):
    """ Use 1 - exp(-x/k) to map positive num into [0, 1) """
    if orig_val <= 0: return 0.0
    return float(1 - np.exp(-orig_val / scale_k))


def normalize_power(orig_val, power_val):
    return orig_val ** power_val


def safe_rmtree(path):
    path = Path(path)

    if not path.exists():
        return

    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    if len(path.parts) <= 2:
        raise ValueError(f"Refusing to delete suspicious path: {path}")

    shutil.rmtree(path)
    
    
def extract_json_dict(json_str: str):
    json_blocks = re.findall(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
    result_json = {}
    for block in json_blocks:
        try:
            result_json = json.loads(block.strip())
            break
        except json.JSONDecodeError as e:
            result_json = {}
    return result_json