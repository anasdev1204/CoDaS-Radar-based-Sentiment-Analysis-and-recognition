import argparse
import json
import os
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

from models.cnn import run_cnn, run_cnn_louo
from models.rnn import run_rnn, run_rnn_louo


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


def build_base_tag(data: str, sensor_tag: str, window: int, overlap: int, event_tag: str = "") -> str:
    if data == "IMU":
        base = f"{sensor_tag}_win{window}s_overlap{overlap}"
    else:
        base = f"win{window}s_overlap{overlap}"
    return f"{base}_{event_tag}" if event_tag else base


def data_dirs(data: str):
    processed = Path("processed") / data.lower()
    return {
        "processed": processed,
        "windows": processed / "windows",
        "rp": processed / "RP",
        "feats": processed / "feats",
    }


def build_paths(
    data: str,
    model: str,
    repr_name: str,
    sensor_tag: str,
    window: int,
    overlap: int,
    img_size: int = 0,
    event_tag: str = "",
):
    base = build_base_tag(data, sensor_tag, window, overlap, event_tag=event_tag)
    dirs = data_dirs(data)
    if model == "CNN":
        suffix = f"_img{img_size}" if img_size > 0 else ""
        x = dirs["rp"] / f"X_{repr_name}_{base}{suffix}.npy"
        y = dirs["rp"] / f"y_{repr_name}_{base}{suffix}.npy"
        u = dirs["rp"] / f"users_{repr_name}_{base}{suffix}.npy"
    elif model == "RNN":
        x = dirs["windows"] / f"X_{base}.npy"
        y = dirs["windows"] / f"y_{base}.npy"
        u = dirs["windows"] / f"users_{base}.npy"
    else:
        raise ValueError("runner currently supports only cnn/rnn")
    return x, y, u, base


def parse_list_int(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_list_float(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_list_str(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def sample_cfg(space: dict, mode: str):
    if mode == "grid":
        # handled outside
        raise RuntimeError("grid mode does not use sample_cfg")
    cfg = {}
    for k, vals in space.items():
        cfg[k] = random.choice(vals)
    return cfg


def grid_cfgs(space: dict):
    keys = list(space.keys())
    cfgs = [{}]
    for k in keys:
        new_cfgs = []
        for c in cfgs:
            for v in space[k]:
                c2 = dict(c)
                c2[k] = v
                new_cfgs.append(c2)
        cfgs = new_cfgs
    return cfgs


def run_one(model: str, louo: bool, x_path: Path, y_path: Path, u_path: Path, cfg: dict):
    if model == "CNN":
        if louo:
            return run_cnn_louo(
                str(x_path),
                str(y_path),
                str(u_path),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                conv1_channels=cfg["conv1"],
                conv2_channels=cfg["conv2"],
                dropout=cfg["dropout"],
                fc_hidden=cfg["fc_hidden"],
                arch=cfg["arch"],
                return_metrics=True,
            )
        return run_cnn(
            str(x_path),
            str(y_path),
            str(u_path),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            conv1_channels=cfg["conv1"],
            conv2_channels=cfg["conv2"],
            dropout=cfg["dropout"],
            fc_hidden=cfg["fc_hidden"],
            arch=cfg["arch"],
            return_metrics=True,
        )

    if model == "RNN":
        if louo:
            return run_rnn_louo(
                str(x_path),
                str(y_path),
                str(u_path),
                arch=cfg["arch"],
                hidden_dim=cfg["hidden_dim"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                dropout=cfg["dropout"],
                num_layers=cfg["num_layers"],
                return_metrics=True,
            )
        return run_rnn(
            str(x_path),
            str(y_path),
            str(u_path),
            arch=cfg["arch"],
            hidden_dim=cfg["hidden_dim"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            dropout=cfg["dropout"],
            num_layers=cfg["num_layers"],
            return_metrics=True,
        )
    raise ValueError("unsupported model")


def parse_args():
    p = argparse.ArgumentParser(description="run cnn/rnn experiments")
    p.add_argument("--data", type=str, required=True, choices=["IMU", "INFRARED", "RADAR", "LORA"])
    p.add_argument("--model", type=str, required=True, choices=["CNN", "RNN"])
    p.add_argument("--louo", type=str, default="True", choices=["True", "False"])
    p.add_argument("--sensor-tag", type=str, default="IMU", help="for imu merged use imu, for single sensor e.g. chest")
    p.add_argument("--event-tag", type=str, default="", help="optional event suffix in feature filenames")
    p.add_argument("--durations", type=int, nargs="+", default=[6])
    p.add_argument("--overlaps", type=int, nargs="+", default=[0])
    p.add_argument("--reprs", type=str, nargs="+", default=["RP"], help="used only for cnn")
    p.add_argument("--img-size", type=int, default=0,
                   help="if >0, use transformed cnn files with suffix _img{size}")
    p.add_argument("--archs", type=str, nargs="+", default=["baseline"], choices=["baseline", "residual", "resnet", "skip_concat"],
                   help="cnn architectures to try")
    p.add_argument("--search", type=str, default="random", choices=["random", "grid"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=str, default="res/tuning_results.csv")
    p.add_argument("--out-best", type=str, default="res/tuning_best.json")
    p.add_argument(
        "--best-by",
        type=str,
        default="combined_acc",
        choices=["acc", "window_acc", "user_acc", "combined_acc"],
    )

    p.add_argument("--epochs", type=str, default="6,10")
    p.add_argument("--batch-sizes", type=str, default="16,32")
    p.add_argument("--lrs", type=str, default="0.001,0.0005")
    p.add_argument("--dropouts", type=str, default="0.3,0.5")

    p.add_argument("--conv1", type=str, default="16,32")
    p.add_argument("--conv2", type=str, default="32,64")
    p.add_argument("--fc-hidden", type=str, default="64,128")

    p.add_argument("--hidden-dims", type=str, default="32,64,128")
    p.add_argument("--num-layers", type=str, default="1,2")
    p.add_argument(
        "--rnn-archs",
        type=str,
        nargs="+",
        default=["lstm"],
        choices=["lstm", "gru", "bilstm"],
        help="rnn architectures to try",
    )

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    louo = args.louo.lower() == "true"
    log("run_models.py started")
    log(
        f"config: data={args.data}, model={args.model}, louo={louo}, "
        f"search={args.search}, trials={args.trials}, seed={args.seed}"
    )

    common = {
        "epochs": parse_list_int(args.epochs),
        "batch_size": parse_list_int(args.batch_sizes),
        "lr": parse_list_float(args.lrs),
        "dropout": parse_list_float(args.dropouts),
    }
    if args.model == "CNN":
        space = {
            **common,
            "arch": args.archs,
            "conv1": parse_list_int(args.conv1),
            "conv2": parse_list_int(args.conv2),
            "fc_hidden": parse_list_int(args.fc_hidden),
        }
    else:
        space = {
            **common,
            "arch": args.rnn_archs,
            "hidden_dim": parse_list_int(args.hidden_dims),
            "num_layers": parse_list_int(args.num_layers),
        }

    fixed_tasks = []
    for d in args.durations:
        for ov in args.overlaps:
            if args.model == "CNN":
                for r in args.reprs:
                    fixed_tasks.append((d, ov, r))
            else:
                fixed_tasks.append((d, ov, "NA"))

    rows = []
    for window, overlap, repr_name in fixed_tasks:
        x_path, y_path, u_path, base = build_paths(
            args.data,
            args.model,
            repr_name,
            args.sensor_tag,
            window,
            overlap,
            args.img_size,
            event_tag=args.event_tag,
        )
        log(f"task: base={base}, repr={repr_name}")
        log(f"paths: X={x_path}, y={y_path}, users={u_path}")
        if not (x_path.exists() and y_path.exists() and u_path.exists()):
            log(f"[skip] missing files for {base} repr={repr_name}")
            continue

        if args.search == "grid":
            cfgs = grid_cfgs(space)
        else:
            cfgs = [sample_cfg(space, "random") for _ in range(args.trials)]
        log(f"total trials for this task: {len(cfgs)}")

        for i, cfg in enumerate(cfgs):
            log(f"[{args.model}] {base} repr={repr_name} trial={i+1}/{len(cfgs)} cfg={cfg}")
            try:
                metrics = run_one(args.model, louo, x_path, y_path, u_path, cfg)
                if isinstance(metrics, dict):
                    window_acc = float(metrics.get("acc", np.nan))
                    louo_acc_std = metrics.get("louo_acc_std", np.nan)
                    user_acc = metrics.get("user_acc", np.nan)
                    user_report = metrics.get("user_report", None)
                    per_user_window_acc = metrics.get("per_user_window_acc", None)
                    class_balanced_acc = metrics.get("balanced_acc", np.nan)
                    acc = class_balanced_acc if not pd.isna(class_balanced_acc) else window_acc
                else:
                    window_acc = float(metrics)
                    louo_acc_std = np.nan
                    user_acc = np.nan
                    user_report = None
                    per_user_window_acc = None
                    class_balanced_acc = np.nan
                    acc = window_acc
                combined_acc = _combined_acc(acc, user_acc)
                log(
                    f"[ok] trial={i+1}/{len(cfgs)} "
                    f"acc={acc:.4f}, window_acc={window_acc}, louo_acc_std={louo_acc_std}, "
                    f"user_acc={user_acc}, "
                    f"combined_acc={combined_acc}"
                )
                row = {
                    "data": args.data,
                    "model": args.model,
                    "louo": louo,
                    "repr": repr_name,
                    "window": window,
                    "overlap": overlap,
                    "base_tag": base,
                    "acc": acc,
                    "window_acc": window_acc,
                    "louo_acc_std": louo_acc_std,
                    "user_acc": user_acc,
                    "combined_acc": combined_acc,
                    "user_report_json": json.dumps(user_report, ensure_ascii=False) if user_report is not None else "",
                    "per_user_window_acc_json": json.dumps(per_user_window_acc, ensure_ascii=False) if per_user_window_acc is not None else "",
                    **cfg,
                }
                rows.append(row)
            except Exception as e:
                log(f"[fail] trial={i+1}/{len(cfgs)} error={e}")
                rows.append(
                    {
                        "data": args.data,
                        "model": args.model,
                        "louo": louo,
                        "repr": repr_name,
                        "window": window,
                        "overlap": overlap,
                        "base_tag": base,
                        "acc": np.nan,
                        "window_acc": np.nan,
                        "louo_acc_std": np.nan,
                        "user_acc": np.nan,
                        "combined_acc": np.nan,
                        "error": str(e),
                        **cfg,
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
    else:
        # keep a stable schema even when all tasks were skipped (e.g., missing files).
        df = pd.DataFrame(
            columns=[
                "data",
                "model",
                "louo",
                "repr",
                "window",
                "overlap",
                "base_tag",
                "acc",
                "window_acc",
                "louo_acc_std",
                "user_acc",
                "combined_acc",
                "user_report_json",
                "per_user_window_acc_json",
                "error",
            ]
        )
    df.to_csv(out_csv, index=False)
    log(f"saved tuning results: {out_csv}")

    best_col = args.best_by if args.best_by in df.columns else "acc"
    valid = df.dropna(subset=[best_col]) if best_col in df.columns else pd.DataFrame()
    if len(valid) == 0 and best_col != "acc" and "acc" in df.columns:
        best_col = "acc"
        valid = df.dropna(subset=[best_col])
    if len(valid):
        best = valid.sort_values(best_col, ascending=False).iloc[0].to_dict()
        out_best = Path(args.out_best)
        out_best.parent.mkdir(parents=True, exist_ok=True)
        out_best.write_text(json.dumps(best, indent=2), encoding="utf-8")
        log(f"best {best_col}={best[best_col]:.4f}, saved: {out_best}")
    else:
        log("no successful runs")
    log("run_models.py finished")


if __name__ == "__main__":
    main()
