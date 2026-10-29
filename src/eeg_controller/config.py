"""Configuration loading and validation.

All processing/model/controller parameters live in a JSON config file
(default: ``configs/default.json``). The parsed, validated structure is
passed around as a frozen :class:`Config` dataclass so no module silently
falls back to hardcoded parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the configuration is missing values or inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    lr_imagery_runs: tuple[int, ...]
    dev_runs: tuple[int, ...]
    final_runs: tuple[int, ...]
    label_map: dict[str, int]
    rest_description: str
    label_names: dict[int, str]


@dataclass(frozen=True)
class PreprocessConfig:
    channels: tuple[str, ...]
    sfreq_hz: float
    bandpass_hz: tuple[float, float]
    filter_order: int
    filtering: str
    settling_s: float
    epoch_start_s: float
    epoch_end_s: float
    baseline_correction: Any


@dataclass(frozen=True)
class ModelConfig:
    csp_components: int
    csp_reg: str
    csp_log: bool
    csp_norm_trace: bool
    lda_solver: str
    lda_shrinkage: Any


@dataclass(frozen=True)
class ControllerConfig:
    threshold: float
    cursor_step: float
    decision_delay_s: float


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    controller: ControllerConfig

    @property
    def samples_per_epoch(self) -> int:
        """Samples in the half-open epoch window at the configured sfreq."""
        span = self.preprocess.epoch_end_s - self.preprocess.epoch_start_s
        return int(round(span * self.preprocess.sfreq_hz))


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required config key {key!r} in [{ctx}]")
    return d[key]


def parse_config(raw: dict[str, Any]) -> Config:
    """Validate a raw config dictionary and return a frozen Config."""
    data = _require(raw, "data", "root")
    lr_runs = tuple(_require(data, "lr_imagery_runs", "data"))
    if not set(lr_runs) == {4, 8, 12}:
        raise ConfigError(
            "data.lr_imagery_runs must be exactly [4, 8, 12] "
            "(left/right imagery). Runs 6/10/14 are hands/feet imagery and "
            "must not be relabeled left/right."
        )
    dev_runs = tuple(_require(data, "dev_runs", "data"))
    final_runs = tuple(_require(data, "final_runs", "data"))
    if set(dev_runs) & set(final_runs):
        raise ConfigError("dev_runs and final_runs must not share runs")
    if not set(dev_runs) | set(final_runs) <= set(lr_runs):
        raise ConfigError("dev/final runs must be a subset of lr_imagery_runs")
    if dev_runs != (4, 8) or final_runs != (12,):
        raise ConfigError("fixed split requires dev_runs [4, 8] and final_runs [12]")
    label_map = dict(_require(data, "label_map", "data"))
    if label_map != {"T1": 0, "T2": 1}:
        raise ConfigError(
            "data.label_map must map T1->0 (left) and T2->1 (right) "
            "explicitly; annotation IDs assigned by MNE must not be relied on"
        )
    rest = _require(data, "rest_description", "data")
    label_names_raw = dict(_require(data, "label_names", "data"))
    label_names = {int(k): v for k, v in label_names_raw.items()}
    if label_names != {0: "left", 1: "right"} or rest != "T0":
        raise ConfigError("labels must be 0=left, 1=right, with T0 rest")

    pre = _require(raw, "preprocess", "root")
    channels = tuple(_require(pre, "channels", "preprocess"))
    if len(set(channels)) != len(channels):
        raise ConfigError("preprocess.channels must not contain duplicates")
    if not {"C3", "Cz", "C4"} <= set(channels):
        raise ConfigError("preprocess.channels must include C3, Cz, C4")
    sfreq = float(_require(pre, "sfreq_hz", "preprocess"))
    band = tuple(_require(pre, "bandpass_hz", "preprocess"))
    if len(band) != 2 or not (0 < band[0] < band[1] < sfreq / 2):
        raise ConfigError(
            "preprocess.bandpass_hz must satisfy 0 < low < high < Nyquist"
        )
    filtering = _require(pre, "filtering", "preprocess")
    if filtering != "sosfilt_causal_forward_only":
        raise ConfigError(
            "preprocess.filtering must be 'sosfilt_causal_forward_only'; "
            "forward-backward filtering is not allowed for causal replay"
        )
    start_s = float(_require(pre, "epoch_start_s", "preprocess"))
    end_s = float(_require(pre, "epoch_end_s", "preprocess"))
    if not (0 <= start_s < end_s):
        raise ConfigError("epoch window must satisfy 0 <= start < end")
    if not all(math.isfinite(x) for x in (sfreq, start_s, end_s)):
        raise ConfigError("sampling rate and epoch boundaries must be finite")
    if _require(pre, "baseline_correction", "preprocess") is not None:
        raise ConfigError("baseline_correction must be null")
    settling = float(_require(pre, "settling_s", "preprocess"))
    if not math.isfinite(settling) or settling < 0:
        raise ConfigError("settling_s must be finite and non-negative")
    if int(pre["filter_order"]) < 1:
        raise ConfigError("filter_order must be positive")

    model = _require(raw, "model", "root")
    csp_components = int(_require(model, "csp_components", "model"))
    if not 1 <= csp_components <= len(channels):
        raise ConfigError("CSP components must be between 1 and the channel count")

    ctrl = _require(raw, "controller", "root")
    threshold = float(_require(ctrl, "threshold", "controller"))
    if not (0.5 <= threshold <= 1.0):
        raise ConfigError("controller.threshold must be within [0.5, 1.0]")
    delay = float(_require(ctrl, "decision_delay_s", "controller"))
    if delay != end_s:
        raise ConfigError("decision_delay_s must equal epoch_end_s; no early decisions")
    step = float(_require(ctrl, "cursor_step", "controller"))
    if not math.isfinite(step) or step <= 0:
        raise ConfigError("cursor_step must be finite and positive")

    return Config(
        seed=int(_require(raw, "seed", "root")),
        data=DataConfig(
            lr_imagery_runs=lr_runs,
            dev_runs=dev_runs,
            final_runs=final_runs,
            label_map=label_map,
            rest_description=str(rest),
            label_names=label_names,
        ),
        preprocess=PreprocessConfig(
            channels=channels,
            sfreq_hz=sfreq,
            bandpass_hz=(float(band[0]), float(band[1])),
            filter_order=int(_require(pre, "filter_order", "preprocess")),
            filtering=str(filtering),
            settling_s=float(_require(pre, "settling_s", "preprocess")),
            epoch_start_s=start_s,
            epoch_end_s=end_s,
            baseline_correction=_require(pre, "baseline_correction", "preprocess"),
        ),
        model=ModelConfig(
            csp_components=csp_components,
            csp_reg=_require(model, "csp_reg", "model"),
            csp_log=bool(_require(model, "csp_log", "model")),
            csp_norm_trace=bool(_require(model, "csp_norm_trace", "model")),
            lda_solver=_require(model, "lda_solver", "model"),
            lda_shrinkage=_require(model, "lda_shrinkage", "model"),
        ),
        controller=ControllerConfig(
            threshold=threshold,
            cursor_step=float(_require(ctrl, "cursor_step", "controller")),
            decision_delay_s=float(_require(ctrl, "decision_delay_s", "controller")),
        ),
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a JSON config file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_config(raw)
