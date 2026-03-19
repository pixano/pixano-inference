# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Framework-agnostic GPU detection via nvidia-smi."""

from __future__ import annotations

import logging
import subprocess
from typing import Any


logger = logging.getLogger(__name__)


def detect_num_gpus() -> int:
    """Detect the number of NVIDIA GPUs via nvidia-smi.

    Returns:
        Number of GPUs found, or 0 on any failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0


def get_gpu_details() -> list[dict[str, Any]]:
    """Get GPU name and memory via nvidia-smi.

    Returns:
        List of dicts with ``index``, ``name``, and ``total_memory_gb`` keys.
        Empty list on any failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        details: list[dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                details.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_memory_gb": float(parts[2]) / 1024,
                    }
                )
        return details
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
