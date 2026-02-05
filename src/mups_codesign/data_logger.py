"""
Run directory creation, metadata snapshot, JSONL metrics, TensorBoard logging.
"""

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        return value.numpy()

    return value


def _is_scalar(value: Any) -> bool:
    if isinstance(value, (bool, int, float, np.number)):
        return True

    if isinstance(value, np.ndarray) and value.shape == ():
        return True

    return False


def _to_serializable(value: Any) -> Any:
    value = _to_numpy(value)
    if _is_scalar(value):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    return str(value)


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            records.append(json.loads(line))

    return records


def load_run(run_dir: str, stream: str = "iteration") -> list:
    filename = "metrics.jsonl" if stream == "iteration" else "control_metrics.jsonl"

    return load_jsonl(os.path.join(run_dir, filename))


class DataLogger:
    def __init__(
        self,
        root_dir: str,
        run_name: str = "codesign"
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = timestamp
        self.run_dir = os.path.join(root_dir, f"{run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.control_metrics_path = os.path.join(self.run_dir, "control_metrics.jsonl")
        self.metadata_path = os.path.join(self.run_dir, "metadata.json")

        self._metrics_fp = None
        self._control_fp = None
        self._tb_writer = None

        self._param_names = None

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        if "param_names" in metadata and self._param_names is None:
            self._param_names = list(metadata["param_names"])

        serializable = {}
        for key, value in metadata.items():
            if is_dataclass(value):
                serializable[key] = _to_serializable(asdict(value))
            else:
                serializable[key] = _to_serializable(value)

        with open(self.metadata_path, "w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2, sort_keys=True)

    def log_iteration(
        self,
        iteration: int,
        objective_total: float,
        objective_terms: Dict[str, Any],
        params_value: Any,
        params_normalized: Any,
        grad_terms: Any,
        grad_norm: float,
        best_loss: float,
        best_params: Any,
        extra: Dict[str, Any],
    ) -> None:
        record: Dict[str, Any] = {
            "iteration": int(iteration),
            "wall_time": time.time(),
            "objective/total": float(objective_total),
        }

        record["achieved_best/loss"] = float(best_loss)
        self._add_param_block(record, "achieved_best", best_params)

        record["gradient/grad_norm"] = float(grad_norm)
        self._add_param_block(record, "gradient", grad_terms)

        for name, value in objective_terms.items():
            record[f"objective/{name}"] = _to_serializable(value)

        self._add_param_block(record, "param/value", params_value)
        self._add_param_block(record, "param/normalized", params_normalized)

        for key, value in extra.items():
            record[f"extra/{key}"] = _to_serializable(value)

        self._write_jsonl(record, stream="iteration")
        self._write_tensorboard(record, step=iteration)

    def log_control_step(self, control_iter: int, scalars: Dict[str, Any]) -> None:
        record = {
            "control_iteration": int(control_iter),
            "wall_time": time.time(),
        }
        for key, value in scalars.items():
            record[f"control/{key}"] = _to_serializable(value)

        self._write_jsonl(record, stream="control")

    def close(self) -> None:
        if self._metrics_fp:
            self._metrics_fp.close()
            self._metrics_fp = None

        if self._control_fp:
            self._control_fp.close()
            self._control_fp = None

        if self._tb_writer:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

        # Delete run dir if empty
        if not os.listdir(self.run_dir):
            os.rmdir(self.run_dir)
            print(f"Deleted empty run directory: {self.run_dir}")

    def _add_param_block(self, record: Dict[str, Any], prefix: str, values: Any) -> None:
        values = _to_numpy(values)
        if isinstance(values, np.ndarray):
            record[f"{prefix}/vector"] = values.tolist()

        if self._param_names is None:
            record[prefix] = _to_serializable(values)
            return

        for name, value in zip(self._param_names, values):
            record[f"{prefix}/{name}"] = _to_serializable(value)

    def _get_metrics_fp(self):
        if self._metrics_fp is None:
            self._metrics_fp = open(self.metrics_path, "a", encoding="utf-8")

        return self._metrics_fp

    def _get_control_fp(self):
        if self._control_fp is None:
            self._control_fp = open(self.control_metrics_path, "a", encoding="utf-8")

        return self._control_fp
    
    def _get_tb_writer(self):
        if self._tb_writer is None:
            tb_dir = os.path.join(self.run_dir, "tensorboard")
            self._tb_writer = SummaryWriter(tb_dir)

        return self._tb_writer

    def _write_jsonl(self, record: Dict[str, Any], stream: str) -> None:
        if stream == "iteration":
            fp = self._get_metrics_fp()
        elif stream == "control":
            fp = self._get_control_fp()
        else:
            raise ValueError(f"Unknown stream: {stream}")

        json.dump(record, fp)
        fp.write("\n")
        fp.flush()

    def _write_tensorboard(self, record: Dict[str, Any], step: int) -> None:
        writer = self._get_tb_writer()

        for key, value in record.items():
            if key in ("iteration", "wall_time", "design_iteration", "control_iteration"):
                continue

            if _is_scalar(value):
                writer.add_scalar(key, float(value), global_step=step)

        writer.flush()
