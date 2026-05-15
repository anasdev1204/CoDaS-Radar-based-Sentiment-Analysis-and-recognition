import argparse
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess_utils import _generate_windows, _zscore_per_user
from transforms import METHODS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "lora"))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "processed", "lora"))
WINDOW_DIR = os.path.join(PROCESSED_DIR, "windows")
RP_DIR = os.path.join(PROCESSED_DIR, "RP")
FEATURE_DIR = os.path.join(PROCESSED_DIR, "feats")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(WINDOW_DIR, exist_ok=True)
os.makedirs(RP_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

EMOTION_MAP = {
    "E01": "focus",
    "E02": "distraction",
    "E03": "stress",
    "E04": "relaxation",
    "E05": "depression",
    "E06": "excitement",
}
DEFAULT_EVENTS = ["E03", "E04"]
DEFAULT_SIGNAL = "diff_200Hz.csv"
OUTLIER_METHOD = "clip"
OUTLIER_Z = 6.0


def _infer_fs_from_name(signal_name: str) -> int:
    m = re.search(r"(\\d+)Hz", signal_name)
    if m:
        return int(m.group(1))
    return 200


def _iter_signal_files(data_dir: str, signal_name: str):
    for user in sorted(os.listdir(data_dir)):
        if not user.startswith("U"):
            continue
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue
        for emo in sorted(os.listdir(user_dir)):
            if not emo.startswith("E"):
                continue
            emo_dir = os.path.join(user_dir, emo)
            if not os.path.isdir(emo_dir):
                continue
            for trial in sorted(os.listdir(emo_dir)):
                trial_dir = os.path.join(emo_dir, trial)
                if not os.path.isdir(trial_dir):
                    continue
                fpath = os.path.join(trial_dir, signal_name)
                if os.path.exists(fpath):
                    yield user, emo, trial, fpath


def _load_signal_csv(path: str):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        return None, None
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(v)
    return t[mask], v[mask]


def _resample(t: np.ndarray, v: np.ndarray, target_fs: int):
    if len(t) < 2:
        return None
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        return None
    step = 1.0 / float(target_fs)
    new_t = np.arange(t0, t1, step, dtype=np.float64)
    new_v = np.interp(new_t, t, v)
    return new_t, new_v


def _remove_outliers(v: np.ndarray, method: str = "clip", z_thresh: float = 6.0) -> np.ndarray:
    if method == "none":
        return v
    if len(v) < 5:
        return v
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-8
    z = (v - med) / (1.4826 * mad)
    mask = np.abs(z) > z_thresh
    if not mask.any():
        return v
    if method == "clip":
        v = v.copy()
        lo = med - z_thresh * 1.4826 * mad
        hi = med + z_thresh * 1.4826 * mad
        v = np.clip(v, lo, hi)
        return v
    if method == "median":
        v = v.copy()
        v[mask] = med
        return v
    return v


def process_lora_data(events, signal_name, target_fs, normalization="zscore"):
    print("\n[1] loading LoRa CSV files...")

    rows = []
    for user, emo_id, trial, path in tqdm(list(_iter_signal_files(DATA_DIR, signal_name)), desc="lora files"):
        emotion = EMOTION_MAP.get(emo_id, "unknown")
        if emotion not in events and emo_id not in events:
            continue

        t, v = _load_signal_csv(path)
        if t is None or len(t) == 0:
            continue

        t_res, v_res = _resample(t, v, target_fs)
        if t_res is None:
            continue
        v_res = _remove_outliers(v_res, method=OUTLIER_METHOD, z_thresh=OUTLIER_Z)

        df = pd.DataFrame(
            {
                "frame": np.arange(len(v_res), dtype=int),
                "user_id": user,
                "emotion_id": emo_id,
                "emotion": emotion,
                "value": v_res.astype(np.float32),
            }
        )
        rows.append(df)

    if not rows:
        print("no LoRa data found")
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    if normalization == "zscore":
        combined = _zscore_per_user(combined, ["value"])

    out_path = os.path.join(PROCESSED_DIR, f"lora_{signal_name.replace('.csv','')}_E03_E04.csv")
    combined.to_csv(out_path, index=False)
    print(f"combined data saved to: {out_path}")
    print(f"final shape: {combined.shape}")
    return combined


def _load_combined(signal_name: str):
    path = os.path.join(PROCESSED_DIR, f"lora_{signal_name.replace('.csv','')}_E03_E04.csv")
    if not os.path.exists(path):
        print(f"missing combined file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def segment(combined_df: pd.DataFrame, durations, overlaps, target_fs: int):
    if combined_df.empty:
        print("combined_df is empty, skipping segmentation")
        return

    print("\n[2] starting segmentation...")
    combined_df = combined_df.sort_values(["user_id", "emotion", "frame"]).reset_index(drop=True)
    feat_cols = ["value"]

    for duration_sec in durations:
        window_size = target_fs * duration_sec
        for overlap in overlaps:
            stride = int(window_size * (1 - overlap / 100))

            X, y, users = _generate_windows(combined_df, feat_cols, window_size, stride)

            name = f"win{duration_sec}s_overlap{overlap}"
            np.save(os.path.join(WINDOW_DIR, f"X_{name}.npy"), X)
            np.save(os.path.join(WINDOW_DIR, f"y_{name}.npy"), y)
            np.save(os.path.join(WINDOW_DIR, f"users_{name}.npy"), users)

            print(f"{name} saved. shape: X {X.shape}, y {y.shape}, users {users.shape}")


def transform_windows(durations, overlaps, method="SPEC", image_size=128, resize=False, log_scale=False):
    transform_fn = METHODS.get(method)
    if transform_fn is None:
        raise ValueError(f"unknown transform method: {method}")

    print(f"\\n[3] starting transform: {method}")
    suffix = f"_img{image_size}" if resize and method != "FEAT" else ""

    for duration_sec in durations:
        for overlap in overlaps:
            name = f"win{duration_sec}s_overlap{overlap}"
            X_path = os.path.join(WINDOW_DIR, f"X_{name}.npy")
            y_path = os.path.join(WINDOW_DIR, f"y_{name}.npy")
            users_path = os.path.join(WINDOW_DIR, f"users_{name}.npy")

            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                print(f"missing files for {name}, skipping")
                continue

            print(f"\\ntransforming {name} with {method}...")
            X = np.load(X_path).astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.load(y_path)
            users = np.load(users_path) if os.path.exists(users_path) else None

            transformed = []
            for window in tqdm(X, desc=f"{method} {name}"):
                out = transform_fn(window, image_size=image_size, resize_flag=resize)
                if method == "SPEC" and log_scale:
                    out = np.log1p(out)
                transformed.append(out.astype(np.float32))

            X_out = np.stack(transformed) if transformed else np.empty((0,))
            target_dir = FEATURE_DIR if method == "FEAT" else RP_DIR

            np.save(os.path.join(target_dir, f"X_{method}_{name}{suffix}.npy"), X_out)
            np.save(os.path.join(target_dir, f"y_{method}_{name}{suffix}.npy"), y)
            if users is not None:
                np.save(os.path.join(target_dir, f"users_{method}_{name}{suffix}.npy"), users)

            print(f"saved: X {X_out.shape}, y {y.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="lora preprocessing pipeline")

    parser.add_argument(
        "--run",
        type=str,
        default="clean",
        choices=["clean", "segment", "transform", "everything"],
        help="which stage to run",
    )
    parser.add_argument(
        "--signal",
        type=str,
        default=DEFAULT_SIGNAL,
        choices=["abs_200Hz.csv", "diff_200Hz.csv", "var_20Hz.csv"],
        help="which signal file to use",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=None,
        help="target sampling rate (Hz); default inferred from signal name",
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[6],
        help="window durations (sec)",
    )
    parser.add_argument(
        "--overlaps",
        type=int,
        nargs="+",
        default=[0],
        help="overlap percentages",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="SPEC",
        choices=sorted(METHODS.keys()),
        help="transform method",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="False",
        choices=["True", "False"],
        help="resize images to fixed image_size",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="output image size (H=W)",
    )
    parser.add_argument(
        "--log-scale",
        type=str,
        default="True",
        choices=["True", "False"],
        help="apply log1p to spectrogram output",
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=DEFAULT_EVENTS,
        help="which emotion events to include (e.g., E03 E04)",
    )
    parser.add_argument(
        "--outlier",
        type=str,
        default="clip",
        choices=["clip", "median", "none"],
        help="outlier handling on signal",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=6.0,
        help="z threshold for outlier detection",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resize_flag = args.resize.lower() == "true"
    log_scale = args.log_scale.lower() == "true"
    target_fs = args.target_fs or _infer_fs_from_name(args.signal)
    global OUTLIER_METHOD, OUTLIER_Z
    OUTLIER_METHOD = args.outlier
    OUTLIER_Z = args.outlier_z

    if args.run == "clean":
        process_lora_data(args.events, args.signal, target_fs)

    elif args.run == "segment":
        combined_df = _load_combined(args.signal)
        segment(combined_df, args.durations, args.overlaps, target_fs)

    elif args.run == "transform":
        transform_windows(
            args.durations,
            args.overlaps,
            method=args.method,
            image_size=args.image_size,
            resize=resize_flag,
            log_scale=log_scale,
        )

    elif args.run == "everything":
        combined_df = process_lora_data(args.events, args.signal, target_fs)
        segment(combined_df, args.durations, args.overlaps, target_fs)
        transform_windows(
            args.durations,
            args.overlaps,
            method=args.method,
            image_size=args.image_size,
            resize=resize_flag,
            log_scale=log_scale,
        )


if __name__ == "__main__":
    main()
