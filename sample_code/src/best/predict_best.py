from pathlib import Path
import argparse
from datetime import datetime
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.logreg import _sanitize_features


CONFIG_PATH = ROOT_DIR / "best_models_config.toml"
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "processed"
PREDICTIONS_DIR = ROOT_DIR / "predictions"
DEFAULT_EVENTS = ["E03", "E04"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run best preprocessing and predict with saved models")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "IMU", "INFRARED", "RADAR"],
        help="which data to run",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="True",
        choices=["True", "False"],
        help="run best preprocessing before prediction",
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=DEFAULT_EVENTS,
        help="which emotion events to use",
    )
    parser.add_argument(
        "--use-validation",
        type=str,
        default="False",
        choices=["True", "False"],
        help="predict only on saved validation split",
    )
    return parser.parse_args()


def _load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _selected_data(values: list[str]) -> list[str]:
    if "all" in values:
        return ["IMU", "INFRARED", "RADAR"]
    return values


def _events_suffix(event_ids: list[str]) -> str:
    if sorted(event_ids) == sorted(DEFAULT_EVENTS):
        return ""
    return "_" + "_".join(sorted(e.lower() for e in event_ids))


def _events_tag(event_ids: list[str]) -> str:
    return "_".join(sorted(e.lower() for e in event_ids))


def _feature_base_tag(base_tag: str, events: list[str]) -> str:
    return f"{base_tag}{_events_suffix(events)}"


def _run_preprocess(selected: list[str], events: list[str]) -> None:
    cmd = [
        sys.executable,
        str(SRC_DIR / "best" / "preprocess_best.py"),
        "--data",
        *selected,
        "--events",
        *events,
    ]
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _feature_paths(modality: str, base_tag: str):
    feat_dir = PROCESSED_DIR / modality.lower() / "feats"
    return (
        feat_dir / f"X_FEAT_{base_tag}.npy",
        feat_dir / f"y_FEAT_{base_tag}.npy",
        feat_dir / f"users_FEAT_{base_tag}.npy",
    )


def _model_path(modality: str, kind: str, events: list[str], use_validation: bool) -> Path:
    val_suffix = "_val" if use_validation else ""
    return MODELS_DIR / f"{modality.lower()}_{kind}_best{_events_suffix(events)}{val_suffix}.pkl"


def _load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _prepare_X(X: np.ndarray, scaler) -> np.ndarray:
    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    X = _sanitize_features(X)
    X = scaler.transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=-20.0)
    X = np.clip(X, -20.0, 20.0)
    return X


def _predict_one(modality: str, cfg: dict, run_tag: str, events: list[str], use_validation: bool) -> Path:
    x_path, y_path, users_path = _feature_paths(modality, _feature_base_tag(cfg["base_tag"], events))
    model_path = _model_path(modality, cfg["kind"], events, use_validation)

    if not x_path.exists():
        raise FileNotFoundError(f"missing features: {x_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing model: {model_path}")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path, allow_pickle=True) if y_path.exists() else None
    users = np.load(users_path, allow_pickle=True) if users_path.exists() else None

    bundle = _load_bundle(model_path)
    clf = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    split = bundle.get("split", {})

    if use_validation:
        val_idx = np.asarray(split.get("validation_indices", []), dtype=int)
        if len(val_idx) == 0:
            raise RuntimeError(f"validation split not found in model artifact: {model_path}")
        X = X[val_idx]
        if y is not None:
            y = y[val_idx]
        if users is not None:
            users = users[val_idx]

    X_ready = _prepare_X(X, scaler)
    y_pred_encoded = clf.predict(X_ready)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_ready).max(axis=1)
    else:
        y_prob = np.full(len(y_pred), np.nan, dtype=np.float32)

    rows = {
        "index": np.arange(len(y_pred)),
        "pred_label": y_pred,
        "pred_score": y_prob,
    }
    if y is not None:
        rows["true_label"] = y
    if users is not None:
        rows["user"] = users

    out_df = pd.DataFrame(rows)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    modality_dir = PREDICTIONS_DIR / modality.lower() / _events_tag(events) / run_tag
    modality_dir.mkdir(parents=True, exist_ok=True)
    out_path = modality_dir / "best_predictions.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\n[saved] predictions: {out_path}")
    print(f"samples: {len(out_df)}")

    if y is not None:
        y_true_encoded = label_encoder.transform(y)
        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        bal_acc = balanced_accuracy_score(y_true_encoded, y_pred_encoded)
        print(f"accuracy: {acc:.4f}")
        print(f"balanced accuracy: {bal_acc:.4f}")
        print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_, zero_division=0))

    if y is not None and users is not None:
        user_rows = []
        for user in np.unique(users):
            mask = users == user
            true_vals = y[mask]
            pred_vals = y_pred[mask]
            true_major = pd.Series(true_vals).mode().iloc[0]
            pred_major = pd.Series(pred_vals).mode().iloc[0]
            user_rows.append(
                {
                    "user": user,
                    "true_label": true_major,
                    "pred_label": pred_major,
                    "correct": bool(true_major == pred_major),
                }
            )
        user_df = pd.DataFrame(user_rows)
        user_out = modality_dir / "best_user_predictions.csv"
        user_df.to_csv(user_out, index=False)
        user_acc = user_df["correct"].mean() if len(user_df) else float("nan")
        print(f"user accuracy: {user_acc:.4f}")
        print(f"[saved] user predictions: {user_out}")

    return out_path


def main() -> None:
    args = parse_args()
    config = _load_config()
    selected = _selected_data(args.data)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_validation = args.use_validation.lower() == "true"

    if args.preprocess.lower() == "true":
        _run_preprocess(selected, args.events)

    for modality in selected:
        key = modality.lower()
        section = config[key]
        cfg = {
            "kind": section["kind"],
            "base_tag": section["base_tag"],
        }
        _predict_one(modality, cfg, run_tag, args.events, use_validation)

    summary_dir = PREDICTIONS_DIR / _events_tag(args.events) / run_tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "best_prediction_summary.json"
    summary = {
        "data": selected,
        "run_tag": run_tag,
        "events": list(args.events),
        "use_validation": use_validation,
        "config_file": str(CONFIG_PATH.name),
        "models_dir": str(MODELS_DIR),
        "predictions_dir": str(PREDICTIONS_DIR),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] summary: {summary_path}")


if __name__ == "__main__":
    main()
