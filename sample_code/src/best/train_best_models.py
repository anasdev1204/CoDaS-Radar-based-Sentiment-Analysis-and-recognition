from pathlib import Path
import argparse
import sys


SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.logreg import (
    train_and_save_sklearn_full,
    train_and_save_sklearn_with_validation,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "processed"
CONFIG_PATH = ROOT_DIR / "best_models_config.toml"
DEFAULT_EVENTS = ["E03", "E04"]


def parse_args():
    parser = argparse.ArgumentParser(description="train and save best classical models")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "IMU", "INFRARED", "RADAR"],
        help="which data to train",
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=DEFAULT_EVENTS,
        help="which emotion events these models correspond to",
    )
    parser.add_argument(
        "--use-validation",
        type=str,
        default="False",
        choices=["True", "False"],
        help="train with a holdout validation split instead of full data",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="validation split size",
    )
    return parser.parse_args()


def _load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _build_model(kind: str, params: dict):
    if kind == "svm":
        return lambda: SVC(
            C=float(params["C"]),
            kernel=params["kernel"],
            gamma=params["gamma"],
            probability=True,
        )
    if kind == "rf":
        return lambda: RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
        )
    raise ValueError(f"unsupported model kind: {kind}")


def _feature_paths(modality: str, base_tag: str):
    feat_dir = PROCESSED_DIR / modality.lower() / "feats"
    x_path = feat_dir / f"X_FEAT_{base_tag}.npy"
    y_path = feat_dir / f"y_FEAT_{base_tag}.npy"
    u_path = feat_dir / f"users_FEAT_{base_tag}.npy"
    return x_path, y_path, u_path


def _events_suffix(event_ids: list[str]) -> str:
    if sorted(event_ids) == sorted(DEFAULT_EVENTS):
        return ""
    return "_" + "_".join(sorted(e.lower() for e in event_ids))


def _feature_base_tag(base_tag: str, events: list[str]) -> str:
    return f"{base_tag}{_events_suffix(events)}"


def train_one(modality: str, cfg: dict, events: list[str], use_validation: bool, val_size: float):
    x_path, y_path, u_path = _feature_paths(modality, _feature_base_tag(cfg["base_tag"], events))
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"missing feature files for {modality}: {x_path} / {y_path}"
        )

    val_suffix = "_val" if use_validation else ""
    save_path = MODELS_DIR / f"{modality.lower()}_{cfg['kind']}_best{_events_suffix(events)}{val_suffix}.pkl"
    if use_validation:
        train_and_save_sklearn_with_validation(
            X_path=str(x_path),
            y_path=str(y_path),
            users_path=str(u_path),
            save_path=str(save_path),
            model_builder=_build_model(cfg["kind"], cfg["params"]),
            metadata=cfg["metadata"],
            oversample=True,
            val_size=val_size,
        )
    else:
        train_and_save_sklearn_full(
            X_path=str(x_path),
            y_path=str(y_path),
            save_path=str(save_path),
            model_builder=_build_model(cfg["kind"], cfg["params"]),
            metadata=cfg["metadata"],
            oversample=True,
        )
    return save_path


def main():
    args = parse_args()
    config = _load_config()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    use_validation = args.use_validation.lower() == "true"
    selected = ["imu", "infrared", "radar"] if "all" in args.data else [x.lower() for x in args.data]
    for key in selected:
        section = config[key]
        cfg = {
            "kind": section["kind"],
            "base_tag": section["base_tag"],
            "params": dict(section["params"]),
            "metadata": {
                "modality": section["modality"],
                "selection_rule": config["selection_rule"],
                "window": section["window"],
                "overlap": section["overlap"],
                "classifier": section["kind"],
                "params": dict(section["params"]),
                "metrics": dict(section["metrics"]),
                "source": config["source"],
                "config_file": str(CONFIG_PATH.name),
                "events": list(args.events),
                "use_validation": use_validation,
                "val_size": float(args.val_size),
            },
        }
        modality = section["modality"]
        save_path = train_one(modality, cfg, args.events, use_validation, args.val_size)
        print(f"[saved] {modality}: {save_path}")


if __name__ == "__main__":
    main()
