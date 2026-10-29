"""Model pipeline: CSP fitted on training epochs only + LDA, save/load.

CSP lives inside a scikit-learn Pipeline so spatial filters are learned
from training data exclusively (guide section 6). Probability columns are
mapped through ``model.classes_`` — never assumed to be [left, right].
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from .config import Config
from .data import hash_file


def build_pipeline(cfg: Config) -> Pipeline:
    """Create the fixed CSP+LDA pipeline from config parameters."""
    m = cfg.model
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=m.csp_components,
                    reg=m.csp_reg,
                    log=m.csp_log,
                    norm_trace=m.csp_norm_trace,
                ),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver=m.lda_solver, shrinkage=m.lda_shrinkage
                ),
            ),
        ]
    )


def predict_with_scores(
    model: Pipeline, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predicted_label, p_left, p_right) using model.classes_.

    p_left is the model probability of class 0 (left) regardless of the
    internal column order in ``predict_proba``.
    """
    preds = model.predict(X)
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    left_col = int(np.where(classes == 0)[0][0])
    right_col = int(np.where(classes == 1)[0][0])
    p_left = proba[:, left_col]
    p_right = proba[:, right_col]
    return np.asarray(preds, dtype=np.int64), p_left, p_right


def save_model(
    model: Pipeline,
    metadata: dict,
    model_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Persist a fitted pipeline (joblib) and provenance metadata (JSON).

    Only project-owned artifacts produced by this module should be loaded
    with :func:`load_model`; untrusted files must never be loaded.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = dict(metadata)
    metadata["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def load_model(model_path: str | Path) -> Pipeline:
    """Load a project-owned fitted pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"model artifact not found: {path}; run scripts/train.py first"
        )
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"{path} does not contain a sklearn Pipeline")
    return model


def provenance_metadata(
    cfg: Config,
    subject_id: int,
    fit_runs: list[int],
    trial_table_path: str | Path,
    seed: int,
) -> dict:
    """Assemble provenance: versions, hashes, contract, class mapping."""
    import pandas as pd
    import scipy
    import sklearn

    from . import __version__

    hashes: dict[str, str] = {}
    tt_path = Path(trial_table_path)
    if tt_path.exists():
        hashes[tt_path.name] = hash_file(tt_path)
        tt = pd.read_csv(tt_path)
        hashes.update(
            {
                name: sha
                for name, sha in zip(tt["edf_file"], tt["edf_sha256"])
                if name not in hashes
            }
        )
    return {
        "subject_id": subject_id,
        "fit_runs": fit_runs,
        "channels": list(cfg.preprocess.channels),
        "channel_order_note": "order is significant; do not reorder",
        "sfreq_hz": cfg.preprocess.sfreq_hz,
        "filter": {
            "type": "butterworth_bandpass_sos_causal",
            "bandpass_hz": list(cfg.preprocess.bandpass_hz),
            "order": cfg.preprocess.filter_order,
            "implementation": "scipy.signal.sosfilt, forward-only",
        },
        "epoch": {
            "start_s": cfg.preprocess.epoch_start_s,
            "end_s": cfg.preprocess.epoch_end_s,
            "samples": cfg.samples_per_epoch,
            "baseline_correction": cfg.preprocess.baseline_correction,
        },
        "class_mapping": {"0": "left (T1)", "1": "right (T2)"},
        "label_source": "explicit annotation description mapping (T1->0, T2->1)",
        "seed": seed,
        "package_versions": {
            "python": sys.version.split()[0],
            "eeg_controller": __version__,
            "mne": __import__("mne").__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "data_file_hashes": hashes,
    }
