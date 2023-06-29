# monitoring.py
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

def log_data(request_id: str, input_data: dict, output_data: dict, timestamp, log_dir: str) -> None:

    os.makedirs(log_dir, exist_ok=True)
    
    log_data = {
        "request_id": request_id,
        "input": input_data,
        "output": output_data,
        "timestamp": timestamp,
    }

    log_file = Path(log_dir) / f"{request_id}.json"
    print(f"Logging data to {log_file}")

    with open(log_file, "w") as f:
        json.dump(log_data, f)