import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.logreg import (
    run_logreg,
    run_logreg_louo,
    run_rf,
    run_rf_louo,
    run_svm,
    run_svm_louo,
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _combined_acc(acc, user_acc):
    if pd.isna(acc) or pd.isna(user_acc):
        return np.nan
    denom = acc + user_acc
    if denom <= 0:
        return 0.0
    return float(2.0 * acc * user_acc / denom)


def parse_list_int(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_str(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_base_tag(data: str, sensor_tag: str, window: int, overlap: int, event_tag: str = "") -> str:
    if data == "IMU":
        base = f"{sensor_tag}_win{window}s_overlap{overlap}"
    else:
        base = f"win{window}s_overlap{overlap}"
    return f"{base}_{event_tag}" if event_tag else base


def build_feat_paths(data: str, sensor_tag: str, window: int, overlap: int, event_tag: str = ""):
    base = build_base_tag(data, sensor_tag, window, overlap, event_tag=event_tag)
    feat_dir = Path("processed") / data.lower() / "feats"
    return (
        feat_dir / f"X_FEAT_{base}.npy",
        feat_dir / f"y_FEAT_{base}.npy",
        feat_dir / f"users_FEAT_{base}.npy",
        base,
    )


def sample_random_cfg(args, rng: random.Random):
    arch = rng.choice(args.classifiers)
    cfg = {"arch": arch}
    if arch == "logreg":
        cfg.update(
            {
                "logreg_c": rng.choice(parse_list_float(args.logreg_c)),
                "logreg_solver": rng.choice(parse_list_str(args.logreg_solver)),
                "logreg_max_iter": rng.choice(parse_list_int(args.logreg_max_iter)),
            }
        )
    elif arch == "svm":
        cfg.update(
            {
                "svm_c": rng.choice(parse_list_float(args.svm_c)),
                "svm_kernel": rng.choice(parse_list_str(args.svm_kernel)),
                "svm_gamma": rng.choice(parse_list_str(args.svm_gamma)),
            }
        )
    else:
        cfg.update(
            {
                "rf_n_estimators": rng.choice(parse_list_int(args.rf_n_estimators)),
                "rf_max_depth": rng.choice(parse_list_int(args.rf_max_depth)),
                "rf_min_samples_split": rng.choice(parse_list_int(args.rf_min_samples_split)),
                "rf_min_samples_leaf": rng.choice(parse_list_int(args.rf_min_samples_leaf)),
            }
        )
    return cfg


def grid_cfgs(args):
    cfgs = []
    for arch in args.classifiers:
        if arch == "logreg":
            for c in parse_list_float(args.logreg_c):
                for solver in parse_list_str(args.logreg_solver):
                    for iters in parse_list_int(args.logreg_max_iter):
                        cfgs.append(
                            {
                                "arch": arch,
                                "logreg_c": c,
                                "logreg_solver": solver,
                                "logreg_max_iter": iters,
                            }
                        )
        elif arch == "svm":
            for c in parse_list_float(args.svm_c):
                for kernel in parse_list_str(args.svm_kernel):
                    for gamma in parse_list_str(args.svm_gamma):
                        cfgs.append(
                            {
                                "arch": arch,
                                "svm_c": c,
                                "svm_kernel": kernel,
                                "svm_gamma": gamma,
                            }
                        )
        elif arch == "rf":
            for n in parse_list_int(args.rf_n_estimators):
                for depth in parse_list_int(args.rf_max_depth):
                    for split in parse_list_int(args.rf_min_samples_split):
                        for leaf in parse_list_int(args.rf_min_samples_leaf):
                            cfgs.append(
                                {
                                    "arch": arch,
                                    "rf_n_estimators": n,
                                    "rf_max_depth": depth,
                                    "rf_min_samples_split": split,
                                    "rf_min_samples_leaf": leaf,
                                }
                            )
    return cfgs


def run_one(cfg: dict, louo: bool, x_path: Path, y_path: Path, users_path: Path):
    arch = cfg["arch"]
    if arch == "logreg":
        if louo:
            return run_logreg_louo(
                str(x_path),
                str(y_path),
                str(users_path),
                C=cfg["logreg_c"],
                solver=cfg["logreg_solver"],
                max_iter=cfg["logreg_max_iter"],
                return_metrics=True,
            )
        return run_logreg(
            str(x_path),
            str(y_path),
            str(users_path),
            C=cfg["logreg_c"],
            solver=cfg["logreg_solver"],
            max_iter=cfg["logreg_max_iter"],
            return_metrics=True,
        )
    if arch == "svm":
        if louo:
            return run_svm_louo(
                str(x_path),
                str(y_path),
                str(users_path),
                C=cfg["svm_c"],
                kernel=cfg["svm_kernel"],
                gamma=cfg["svm_gamma"],
                return_metrics=True,
            )
        return run_svm(
            str(x_path),
            str(y_path),
            str(users_path),
            C=cfg["svm_c"],
            kernel=cfg["svm_kernel"],
            gamma=cfg["svm_gamma"],
            return_metrics=True,
        )
    if louo:
        return run_rf_louo(
            str(x_path),
            str(y_path),
            str(users_path),
            n_estimators=cfg["rf_n_estimators"],
            max_depth=None if cfg["rf_max_depth"] == 0 else cfg["rf_max_depth"],
            min_samples_split=cfg["rf_min_samples_split"],
            min_samples_leaf=cfg["rf_min_samples_leaf"],
            return_metrics=True,
        )
    return run_rf(
        str(x_path),
        str(y_path),
        str(users_path),
        n_estimators=cfg["rf_n_estimators"],
        max_depth=None if cfg["rf_max_depth"] == 0 else cfg["rf_max_depth"],
        min_samples_split=cfg["rf_min_samples_split"],
        min_samples_leaf=cfg["rf_min_samples_leaf"],
        return_metrics=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="run classical classifier experiments (logreg/svm/rf)")
    p.add_argument("--data", type=str, required=True, choices=["IMU", "INFRARED", "RADAR", "LORA"])
    p.add_argument("--louo", type=str, default="True", choices=["True", "False"])
    p.add_argument("--sensor-tag", type=str, default="IMU")
    p.add_argument("--event-tag", type=str, default="", help="optional event suffix in feature filenames")
    p.add_argument("--durations", type=int, nargs="+", default=[6])
    p.add_argument("--overlaps", type=int, nargs="+", default=[0])
    p.add_argument("--classifiers", type=str, nargs="+", default=["logreg", "svm", "rf"])
    p.add_argument("--search", type=str, default="random", choices=["random", "grid"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--best-by",
        type=str,
        default="combined_acc",
        choices=["acc", "window_acc", "user_acc", "combined_acc"],
    )
    p.add_argument("--out-csv", type=str, default="res/classifier_tuning/tuning_results.csv")
    p.add_argument("--out-best", type=str, default="res/classifier_tuning/tuning_best.json")

    p.add_argument("--logreg-c", type=str, default="0.1,1.0,10.0")
    p.add_argument("--logreg-solver", type=str, default="lbfgs,liblinear")
    p.add_argument("--logreg-max-iter", type=str, default="1000,2000")

    p.add_argument("--svm-c", type=str, default="0.5,1.0,2.0")
    p.add_argument("--svm-kernel", type=str, default="rbf,linear")
    p.add_argument("--svm-gamma", type=str, default="scale,auto")

    p.add_argument("--rf-n-estimators", type=str, default="100,200,400")
    p.add_argument("--rf-max-depth", type=str, default="0,10,20")
    p.add_argument("--rf-min-samples-split", type=str, default="2,5")
    p.add_argument("--rf-min-samples-leaf", type=str, default="1,2")
    return p.parse_args()


def main():
    args = parse_args()
    louo = args.louo.lower() == "true"
    rng = random.Random(args.seed)
    log(
        f"run_classifiers started: data={args.data}, louo={louo}, "
        f"search={args.search}, trials={args.trials}, classifiers={args.classifiers}"
    )

    rows = []
    for duration in args.durations:
        for overlap in args.overlaps:
            x_path, y_path, users_path, base = build_feat_paths(
                args.data,
                args.sensor_tag,
                duration,
                overlap,
                event_tag=args.event_tag,
            )
            log(f"task: {base}")
            log(f"paths: X={x_path}, y={y_path}, users={users_path}")
            if not (x_path.exists() and y_path.exists() and users_path.exists()):
                log(f"[skip] missing files for {base}")
                continue

            cfgs = grid_cfgs(args) if args.search == "grid" else [
                sample_random_cfg(args, rng) for _ in range(args.trials)
            ]
            log(f"total trials for task {base}: {len(cfgs)}")

            for i, cfg in enumerate(cfgs, 1):
                log(f"trial {i}/{len(cfgs)} cfg={cfg}")
                try:
                    metrics = run_one(cfg, louo, x_path, y_path, users_path)
                    window_acc = float(metrics.get("acc", np.nan))
                    louo_acc_std = metrics.get("louo_acc_std", np.nan)
                    class_balanced_acc = metrics.get("balanced_acc", np.nan)
                    acc = class_balanced_acc if not pd.isna(class_balanced_acc) else window_acc
                    user_acc = metrics.get("user_acc", np.nan)
                    user_report = metrics.get("user_report", None)
                    per_user_window_acc = metrics.get("per_user_window_acc", None)
                    combined_acc = _combined_acc(acc, user_acc)
                    rows.append(
                        {
                            "data": args.data,
                            "model": "CLASSICAL",
                            "louo": louo,
                            "base_tag": base,
                            "window": duration,
                            "overlap": overlap,
                            "acc": acc,
                            "window_acc": window_acc,
                            "louo_acc_std": louo_acc_std,
                            "user_acc": user_acc,
                            "combined_acc": combined_acc,
                            "user_report_json": json.dumps(user_report, ensure_ascii=False) if user_report is not None else "",
                            "per_user_window_acc_json": json.dumps(per_user_window_acc, ensure_ascii=False) if per_user_window_acc is not None else "",
                            **cfg,
                        }
                    )
                    log(
                        f"[ok] acc={acc}, window_acc={window_acc}, "
                        f"user_acc={user_acc}, "
                        f"combined_acc={combined_acc}"
                    )
                except Exception as e:
                    rows.append(
                        {
                            "data": args.data,
                            "model": "CLASSICAL",
                            "louo": louo,
                            "base_tag": base,
                            "window": duration,
                            "overlap": overlap,
                            "acc": np.nan,
                            "window_acc": np.nan,
                            "louo_acc_std": np.nan,
                            "user_acc": np.nan,
                            "combined_acc": np.nan,
                            "user_report_json": "",
                            "per_user_window_acc_json": "",
                            "error": str(e),
                            **cfg,
                        }
                    )
                    log(f"[fail] {e}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "data",
            "model",
            "louo",
            "base_tag",
            "window",
            "overlap",
            "acc",
            "window_acc",
            "louo_acc_std",
            "user_acc",
            "combined_acc",
            "user_report_json",
            "per_user_window_acc_json",
            "arch",
            "error",
        ]
    )
    df.to_csv(out_csv, index=False)
    log(f"saved run results: {out_csv}")

    best_col = args.best_by if args.best_by in df.columns else "acc"
    valid = df.dropna(subset=[best_col]) if best_col in df.columns else pd.DataFrame()
    if len(valid) == 0 and best_col != "acc" and "acc" in df.columns:
        best_col = "acc"
        valid = df.dropna(subset=[best_col])
    out_best = Path(args.out_best)
    out_best.parent.mkdir(parents=True, exist_ok=True)
    if len(valid):
        best = valid.sort_values(best_col, ascending=False).iloc[0].to_dict()
        out_best.write_text(json.dumps(best, indent=2), encoding="utf-8")
        log(f"best {best_col}={best[best_col]:.4f}, saved: {out_best}")
    else:
        out_best.write_text(json.dumps({"error": "no successful runs"}, indent=2), encoding="utf-8")
        log("no successful runs")


if __name__ == "__main__":
    main()
