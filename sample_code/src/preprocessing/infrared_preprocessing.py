import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess_utils import (
    process_single_file_wide,
    _generate_windows,
    _zscore_per_user,
)
from transforms import METHODS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "infrared"))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "processed", "infrared"))
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
DEFAULT_NORMALIZATION = "zscore"
SAMPLING_RATE = 100


def _event_suffix(event_ids: list[str]) -> str:
    if sorted(event_ids) == sorted(DEFAULT_EVENTS):
        return ""
    return "_" + "_".join(sorted(e.lower() for e in event_ids))


def _combined_path(event_ids: list[str]) -> str:
    tag = "_".join(sorted(event_ids))
    return os.path.join(PROCESSED_DIR, f"infrared_{tag}.csv")


def _select_files(data_dir: str, event_ids: list[str]) -> list[str]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    return sorted([f for f in files if any(e in f for e in event_ids)])


def process_infrared_data(event_ids: list[str], normalization: str) -> pd.DataFrame:
    print("\n[1] combining infrared CSV files...")

    files = _select_files(DATA_DIR, event_ids)
    if not files:
        print("no infrared files found")
        return pd.DataFrame()

    dfs = []
    for i, filename in enumerate(files, 1):
        print(f"processing file {i}/{len(files)}: {filename}")
        file_path = os.path.join(DATA_DIR, filename)
        df = process_single_file_wide(file_path)
        if df.empty:
            continue
        dfs.append(df)

    if not dfs:
        print("no data after processing")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    meta_cols = ["frame", "user_id", "emotion_id", "emotion"]
    feat_cols = [c for c in combined_df.columns if c not in meta_cols]
    if normalization == "zscore":
        combined_df = _zscore_per_user(combined_df, feat_cols)
    elif normalization == "minmax":
        combined_df[feat_cols] = (
            combined_df[feat_cols] - combined_df[feat_cols].min()
        ) / (combined_df[feat_cols].max() - combined_df[feat_cols].min() + 1e-8)
    output_path = _combined_path(event_ids)
    combined_df.to_csv(output_path, index=False)

    print(f"combined data saved to: {output_path}")
    print(f"final shape: {combined_df.shape}")
    return combined_df


def _load_combined_df(event_ids: list[str]) -> pd.DataFrame:
    path = _combined_path(event_ids)
    if not os.path.exists(path):
        print(f"missing combined file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def segment(combined_df: pd.DataFrame, durations, overlaps, event_ids: list[str]):
    if combined_df.empty:
        print("combined_df is empty, skipping segmentation")
        return

    print("\n[2] starting segmentation...")
    combined_df = combined_df.sort_values(["user_id", "emotion", "frame"]).reset_index(drop=True)
    meta_cols = ["frame", "user_id", "emotion_id", "emotion"]
    feat_cols = [c for c in combined_df.columns if c not in meta_cols]

    for duration_sec in durations:
        window_size = SAMPLING_RATE * duration_sec
        for overlap in overlaps:
            stride = int(window_size * (1 - overlap / 100))

            X, y, users = _generate_windows(combined_df, feat_cols, window_size, stride)

            name = f"win{duration_sec}s_overlap{overlap}{_event_suffix(event_ids)}"
            np.save(os.path.join(WINDOW_DIR, f"X_{name}.npy"), X)
            np.save(os.path.join(WINDOW_DIR, f"y_{name}.npy"), y)
            np.save(os.path.join(WINDOW_DIR, f"users_{name}.npy"), users)

            print(f"{name} saved. shape: X {X.shape}, y {y.shape}, users {users.shape}")


def transform_windows(durations, overlaps, event_ids: list[str], method="RP", image_size=128, resize=False):
    if resize and image_size <= 0:
        print(f"[warn] invalid image_size={image_size}, fallback to 128")
        image_size = 128

    transform_fn = METHODS.get(method)
    if transform_fn is None:
        raise ValueError(f"unknown transform method: {method}")

    print(f"\n[3] starting transform: {method}")
    suffix = f"_img{image_size}" if resize and method != "FEAT" else ""

    for duration_sec in durations:
        for overlap in overlaps:
            name = f"win{duration_sec}s_overlap{overlap}{_event_suffix(event_ids)}"
            X_path = os.path.join(WINDOW_DIR, f"X_{name}.npy")
            y_path = os.path.join(WINDOW_DIR, f"y_{name}.npy")
            users_path = os.path.join(WINDOW_DIR, f"users_{name}.npy")

            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                print(f"missing files for {name}, skipping")
                continue

            print(f"\ntransforming {name} with {method}...")
            X = np.load(X_path).astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.load(y_path)
            users = np.load(users_path) if os.path.exists(users_path) else None

            transformed = []
            for window in tqdm(X, desc=f"{method} {name}"):
                out = transform_fn(window, image_size=image_size, resize_flag=resize)
                transformed.append(out.astype(np.float32))

            X_out = np.stack(transformed) if transformed else np.empty((0,))
            target_dir = FEATURE_DIR if method == "FEAT" else RP_DIR

            np.save(os.path.join(target_dir, f"X_{method}_{name}{suffix}.npy"), X_out)
            np.save(os.path.join(target_dir, f"y_{method}_{name}{suffix}.npy"), y)
            if users is not None:
                np.save(os.path.join(target_dir, f"users_{method}_{name}{suffix}.npy"), users)

            print(f"saved: X {X_out.shape}, y {y.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="infrared preprocessing pipeline")

    parser.add_argument(
        "--run",
        type=str,
        default="clean",
        choices=["clean", "segment", "transform", "everything"],
        help="which stage to run",
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
        default="RP",
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
        "--events",
        type=str,
        nargs="+",
        default=DEFAULT_EVENTS,
        help="which emotion events to include (e.g., E01 E02 E03 E04)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    resize_flag = args.resize.lower() == "true"

    if args.run == "clean":
        process_infrared_data(args.events, DEFAULT_NORMALIZATION)

    elif args.run == "segment":
        combined_df = _load_combined_df(args.events)
        segment(combined_df, args.durations, args.overlaps, args.events)

    elif args.run == "transform":
        transform_windows(
            args.durations,
            args.overlaps,
            args.events,
            method=args.method,
            image_size=args.image_size,
            resize=resize_flag,
        )

    elif args.run == "everything":
        combined_df = process_infrared_data(args.events, DEFAULT_NORMALIZATION)
        segment(combined_df, args.durations, args.overlaps, args.events)
        transform_windows(
            args.durations,
            args.overlaps,
            args.events,
            method=args.method,
            image_size=args.image_size,
            resize=resize_flag,
        )


if __name__ == "__main__":
    main()
