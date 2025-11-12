import os
from typing import Tuple


def resolve_num_workers(requested_workers: int) -> Tuple[int, bool]:
    """
    Determine number of DataLoader workers.
    Args:
        requested_workers: >=0 for explicit override, else auto.
    Returns:
        (worker_count, use_workers_flag)
    """
    if requested_workers is not None and requested_workers >= 0:
        workers = requested_workers
    else:
        cpu_cnt = os.cpu_count() or 4
        workers = min(8, max(2, cpu_cnt // 2))
    return workers, workers > 0
