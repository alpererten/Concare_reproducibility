from __future__ import annotations

import json
import os
import pickle
import platform
import random
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
from torch.utils.data import Dataset

# ---------------------------
# Discretizer and Normalizer
# ---------------------------

class Discretizer:
    """
    Merges functionality from the original preprocessing.py Discretizer.
    Output parity is preserved:
      - transform returns (data: np.ndarray, new_header: str)
      - new_header is a comma-joined string identical to the original
      - data layout and mask handling are unchanged
    """

    def __init__(
        self,
        timestep: float = 0.8,
        store_masks: bool = True,
        impute_strategy: str = "zero",
        start_time: str = "zero",
        config_path: str | None = None,
    ) -> None:


        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "data", "discretizer_config.json")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        self._id_to_channel: List[str] = cfg["id_to_channel"]
        self._channel_to_id: Dict[str, int] = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
        self._is_categorical_channel: Dict[str, bool] = cfg["is_categorical_channel"]
        self._possible_values: Dict[str, List[str]] = cfg["possible_values"]
        self._normal_values: Dict[str, Any] = cfg["normal_values"]

        self._header: List[str] = ["Hours"] + self._id_to_channel
        self._timestep: float = timestep
        self._store_masks: bool = store_masks
        self._start_time: str = start_time
        self._impute_strategy: str = impute_strategy
        self._done_count: int = 0
        self._empty_bins_sum: float = 0.0
        self._unused_data_sum: float = 0.0



    # ---------- small helpers ----------

    def _first_time(self, ts: List[float]) -> float:
        if self._start_time == "relative":
            return ts[0]
        if self._start_time == "zero":
            return 0.0
        raise ValueError("start_time is invalid")

    def _layout(self) -> Tuple[List[int], List[int], int]:
        cur = 0
        begins: List[int] = []
        ends: List[int] = []
        for ch in self._id_to_channel:
            begins.append(cur)
            if self._is_categorical_channel[ch]:
                cur += len(self._possible_values[ch])
            else:
                cur += 1
            ends.append(cur)
        return begins, ends, cur

    def _write_value(
        self,
        dst: np.ndarray,
        bin_id: int,
        channel: str,
        value: Any,
        begin_pos: List[int],
    ) -> None:
        cid = self._channel_to_id[channel]
        if self._is_categorical_channel[channel]:
            cats = self._possible_values[channel]
            one = np.zeros((len(cats),))
            one[cats.index(value)] = 1.0
            dst[bin_id, begin_pos[cid] : begin_pos[cid] + len(cats)] = one
        else:
            dst[bin_id, begin_pos[cid]] = float(value)

    def _build_header_parts(self) -> List[str]:
        parts: List[str] = []
        for ch in self._id_to_channel:
            if self._is_categorical_channel[ch]:
                for v in self._possible_values[ch]:
                    parts.append(f"{ch}->{v}")
            else:
                parts.append(ch)
        if self._store_masks:
            for ch in self._id_to_channel:
                parts.append(f"mask->{ch}")
        return parts

    def _impute_forward(
        self,
        mask: np.ndarray,
        original_value: List[List[str]],
        data: np.ndarray,
        begin_pos: List[int],
    ) -> None:
        prev_vals: List[List[Any]] = [[] for _ in range(len(self._id_to_channel))]
        for b in range(mask.shape[0]):
            for ch in self._id_to_channel:
                cid = self._channel_to_id[ch]
                if mask[b, cid] == 1:
                    prev_vals[cid].append(original_value[b][cid])
                else:
                    if self._impute_strategy == "normal_value":
                        v = self._normal_values[ch]
                    else:  # previous
                        v = prev_vals[cid][-1] if prev_vals[cid] else self._normal_values[ch]
                    self._write_value(data, b, ch, v, begin_pos)

    def _impute_backward(
        self,
        mask: np.ndarray,
        original_value: List[List[str]],
        data: np.ndarray,
        begin_pos: List[int],
    ) -> None:
        next_vals: List[List[Any]] = [[] for _ in range(len(self._id_to_channel))]
        for b in range(mask.shape[0] - 1, -1, -1):
            for ch in self._id_to_channel:
                cid = self._channel_to_id[ch]
                if mask[b, cid] == 1:
                    next_vals[cid].append(original_value[b][cid])
                else:
                    v = next_vals[cid][-1] if next_vals[cid] else self._normal_values[ch]
                    self._write_value(data, b, ch, v, begin_pos)


    # ---------- main API----------

    def transform(
        self,
        X: np.ndarray | List[List[Any]],
        header: List[str] | None = None,
        end: float | None = None,
    ) -> Tuple[np.ndarray, str]:
        header = header or self._header
        assert header[0] == "Hours"
        eps = 1e-6

        ts = [float(r[0]) for r in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        first = self._first_time(ts)
        max_hours = (max(ts) - first) if end is None else (end - first)
        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        begin_pos, end_pos, total_feat_len = self._layout()
        data = np.zeros((N_bins, total_feat_len), dtype=float)
        mask = np.zeros((N_bins, len(self._id_to_channel)), dtype=int)
        original_value: List[List[str]] = [["" for _ in self._id_to_channel] for _ in range(N_bins)]

        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                val = row[j]
                if val == "":
                    continue
                ch = header[j]
                cid = self._channel_to_id[ch]

                total_data += 1
                if mask[bin_id, cid] == 1:
                    unused_data += 1
                mask[bin_id, cid] = 1

                self._write_value(data, bin_id, ch, val, begin_pos)
                original_value[bin_id][cid] = val

        if self._impute_strategy not in {"zero", "normal_value", "previous", "next"}:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in {"normal_value", "previous"}:
            self._impute_forward(mask, original_value, data, begin_pos)
        elif self._impute_strategy == "next":
            self._impute_backward(mask, original_value, data, begin_pos)
        # else "zero" keeps zeros that are already in data

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        new_header = ",".join(self._build_header_parts())
        return data, new_header


class Normalizer:
    """
    Minimal Normalizer used by materialize_ram.py:
      - __init__ holds means/stds (set externally)
      - transform applies (x - mean) / std
    """
    def __init__(self, fields: Iterable[int] | None = None) -> None:
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._fields = list(fields) if fields is not None else None

    def _normalize_fields(self, X: np.ndarray, fields: Iterable[int]) -> np.ndarray:
        for col in fields:
            X[:, col] = (X[:, col] - self._means[col]) / self._stds[col]  # type: ignore[index]
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        fields = self._fields or range(X.shape[1])
        return self._normalize_fields(X.copy().astype(float), fields)


class InHospitalMortalityReader(Dataset):
    """
    PyTorch Dataset with legacy Reader wrappers for materialize_ram.py compatibility.
    """

    def __init__(
        self,
        dataset_dir: str,
        listfile: str | None = None,
        period_length: float = 48.0,
    ) -> None:
        super().__init__()
        self._dataset_dir = dataset_dir
        self._period_length = period_length
        self._current_index = 0

        listfile_path = listfile or os.path.join(dataset_dir, "listfile.csv")
        with open(listfile_path, "r") as f:
            lines = f.readlines()

        # first line is header in the original implementation
        self._listfile_header = lines[0]
        data_lines = lines[1:]

        pairs: List[Tuple[str, int]] = []
        for line in data_lines:
            x, y = line.strip().split(",")
            pairs.append((x, int(y)))

        self._records: List[Tuple[str, int]] = pairs

    # --- Helpers and I/O parity ---

    def _read_timeseries(self, ts_filename: str) -> Tuple[np.ndarray, List[str]]:
        rows: List[np.ndarray] = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsf:
            header = tsf.readline().strip().split(",")
            assert header[0] == "Hours"
            for line in tsf:
                mas = line.strip().split(",")
                rows.append(np.array(mas))
        return np.stack(rows), header

    # --- PyTorch Dataset API ---

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self._records):
            raise ValueError("Index must be from 0 inclusive to the number of examples exclusive.")
        name, y = self._records[index]
        X, header = self._read_timeseries(name)
        t = self._period_length
        return {"X": X, "t": t, "y": y, "header": header, "name": name}

    # --- Legacy Reader wrappers expected by materialize_ram.py ---

    def get_number_of_examples(self) -> int:
        return len(self)

    def read_example(self, index: int) -> Dict[str, Any]:
        return self[index]



__all__ = [
    "Discretizer",
    "Normalizer",
    "InHospitalMortalityReader",
]
