import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transforms import *
from preprocess_utils import (
    _generate_windows,
    _zscore_per_user,
)
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "imu"))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "processed", "imu"))
SENSORS_RAW_DIR = os.path.join(PROCESSED_DIR, "sensors")
RESAMPLED_DIR = os.path.join(PROCESSED_DIR, "sensors_resampled")
WINDOW_DIR = os.path.join(PROCESSED_DIR, "windows")
RP_DIR = os.path.join(PROCESSED_DIR, "RP")
FEATURE_DIR = os.path.join(PROCESSED_DIR, "feats")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SENSORS_RAW_DIR, exist_ok=True)
os.makedirs(RESAMPLED_DIR, exist_ok=True)
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
ACTIVE_EVENTS = DEFAULT_EVENTS.copy()
EMOTIONS = [EMOTION_MAP[e] for e in DEFAULT_EVENTS]
TARGET_FS = 100

METHODS = {
    "RP": generate_RP,
    "GAF": generate_GAF,
    "SPEC": generate_Spectrogram,
    "MTF": generate_MTF,
    "LINE": generate_LinePlot,
    "FEAT": generate_features,
}
MERGE_SENSORS = False


def _event_suffix() -> str:
    if sorted(ACTIVE_EVENTS) == sorted(DEFAULT_EVENTS):
        return ""
    return "_" + "_".join(sorted(e.lower() for e in ACTIVE_EVENTS))


def _load_resampled_sensor(resampled_dir, sensor):
    fpath = os.path.join(resampled_dir, f"{sensor}_resampled.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    return pd.read_csv(fpath)


def _build_merged_sensor_df(resampled_dir):
    meta_cols = ["user_id", "emotion_id", "emotion", "frame"]
    merged_df = None

    for sensor in SENSORS:
        df = _load_resampled_sensor(resampled_dir, sensor)
        if df.empty:
            print(f"{sensor}: resampled file not found/empty, skip in merge.")
            continue

        df = df[df["emotion"].isin(EMOTIONS)].copy()
        feat_cols = [
            c
            for c in df.columns
            if any(k in c for k in ["acc_", "gyro_", "magn_"])
        ]
        if not feat_cols:
            continue

        rename_map = {c: f"{sensor}_{c}" for c in feat_cols}
        df = df[meta_cols + feat_cols].rename(columns=rename_map)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on=meta_cols, how="inner")

    if merged_df is None:
        return pd.DataFrame()

    merged_df = merged_df.sort_values(meta_cols).reset_index(drop=True)
    return merged_df

def process_sensor_data():
    print(f"\n[1] combining raw csv files for events: {ACTIVE_EVENTS} ...")

    RX_TIME = re.compile("time", re.I)
    RX_CH = {
        "acc": {ax: re.compile(rf"acc.*{ax}$|accelerometer.*{ax}", re.I) for ax in "xyz"},
        "gyro": {ax: re.compile(rf"gyro.*{ax}$|gyr.*{ax}", re.I) for ax in "xyz"},
        "magn": {ax: re.compile(rf"mag.*{ax}$|magn.*{ax}", re.I) for ax in "xyz"},
    }

    def find_col(cols, rx):
        for c in cols:
            if rx.search(c):
                return c
        return None

    users = sorted([u for u in os.listdir(DATA_DIR) if u.startswith("U")])

    for sensor in SENSORS:
        out_file = os.path.join(SENSORS_RAW_DIR, f"{sensor}_data.csv")
        if os.path.exists(out_file):
            os.remove(out_file)

        for user in tqdm(users, desc=f"{sensor} users"):
            user_dir = os.path.join(DATA_DIR, user)
            emo_dirs = sorted([d for d in os.listdir(user_dir) if d.startswith("E")])

            for emo_dir in emo_dirs:
                emotion_id = emo_dir.split("-")[0]
                emotion = EMOTION_MAP.get(emotion_id)

                if emotion not in EMOTIONS:
                    continue

                root_path = os.path.join(user_dir, emo_dir)
                filepaths = []
                for root, _, files in os.walk(root_path):
                    for fname in files:
                        if fname.startswith(sensor) and fname.lower().endswith(".csv"):
                            filepaths.append(os.path.join(root, fname))

                if not filepaths:
                    continue

                typed_frames = {}

                for fpath in filepaths:
                    df = pd.read_csv(fpath, low_memory=False)
                    cols = list(df.columns)

                    ts_col = find_col(cols, RX_TIME)
                    if ts_col is None:
                        continue

                    for t in ["acc", "gyro", "magn"]:
                        ch = {}
                        for ax in "xyz":
                            col = find_col(cols, RX_CH[t][ax])
                            if col:
                                ch[ax] = col

                        if len(ch) == 3:
                            sub = pd.DataFrame(
                                {
                                    "timestamp": pd.to_numeric(
                                        df[ts_col], errors="coerce"
                                    ),
                                    f"{t}_x": pd.to_numeric(
                                        df[ch["x"]], errors="coerce"
                                    ),
                                    f"{t}_y": pd.to_numeric(
                                        df[ch["y"]], errors="coerce"
                                    ),
                                    f"{t}_z": pd.to_numeric(
                                        df[ch["z"]], errors="coerce"
                                    ),
                                }
                            )
                            typed_frames.setdefault(t, []).append(sub)
                            break

                if not typed_frames:
                    continue

                for t, lst in typed_frames.items():
                    df_t = pd.concat(lst, ignore_index=True)
                    df_t = df_t.dropna(subset=["timestamp"])
                    df_t = df_t.groupby("timestamp", as_index=False).mean()
                    typed_frames[t] = df_t

                merged = None
                for t in ["acc", "gyro", "magn"]:
                    if t in typed_frames:
                        merged = (
                            typed_frames[t]
                            if merged is None
                            else merged.merge(
                                typed_frames[t], on="timestamp", how="outer"
                            )
                        )

                if merged is None or merged.empty:
                    continue

                merged = merged.loc[:, ~merged.columns.duplicated()]

                feat_cols = [
                    c
                    for c in merged.columns
                    if any(k in c for k in ["acc_", "gyro_", "magn_"])
                ]

                merged["timestamp"] = pd.to_numeric(
                    merged["timestamp"], errors="coerce"
                )
                merged[feat_cols] = (
                    merged[feat_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .astype(np.float32)
                )

                merged = merged.dropna(subset=["timestamp"])
                merged = merged.sort_values("timestamp").drop_duplicates("timestamp")

                merged["user_id"] = user
                merged["emotion"] = emotion
                merged["emotion_id"] = emotion_id

                merged = merged[
                    ["user_id", "emotion_id", "emotion", "timestamp"] + feat_cols
                ]

                merged.to_csv(
                    out_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(out_file),
                )

        print(f"{sensor}: raw merged file saved -> {out_file}")

    print("raw sensor processing done")
    return SENSORS_RAW_DIR

def resample_sensors(sensor_dir=SENSORS_RAW_DIR):
    print("\n[2] resampling sensors to "
          f"{TARGET_FS} Hz and saving to {RESAMPLED_DIR}...")

    for fname in tqdm(sorted(os.listdir(sensor_dir)), desc="resampling sensors"):
        if not fname.endswith("_data.csv"):
            continue

        sensor = fname.replace("_data.csv", "")
        df = pd.read_csv(os.path.join(sensor_dir, fname))

        feat_cols = [
            c
            for c in df.columns
            if any(k in c for k in ["acc_", "gyro_", "magn_"])
        ]

        dfs_resampled = []

        for (uid, emo_id, emo), sub in df.groupby(
            ["user_id", "emotion_id", "emotion"], sort=False
        ):
            sub = sub.copy()
            sub["timestamp"] = pd.to_numeric(sub["timestamp"], errors="coerce")
            sub = sub.dropna(subset=["timestamp"])
            if len(sub) < 5:
                continue

            sub = sub.sort_values("timestamp")

            sub[feat_cols] = (
                sub[feat_cols]
                .apply(pd.to_numeric, errors="coerce")
                .astype(np.float32)
            )

            t_min, t_max = sub["timestamp"].min(), sub["timestamp"].max()
            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
                continue

            new_t = np.arange(t_min, t_max, 1.0 / TARGET_FS, dtype=np.float64)
            out = pd.DataFrame({"timestamp": new_t})

            for c in feat_cols:
                out[c] = np.interp(new_t, sub["timestamp"].values, sub[c].values)

            out["user_id"] = uid
            out["emotion_id"] = emo_id
            out["emotion"] = emo
            out["frame"] = np.arange(len(out), dtype=int)

            dfs_resampled.append(out)

        if not dfs_resampled:
            print(f"{sensor}: nothing to resample, skipping.")
            continue

        resampled = pd.concat(dfs_resampled, ignore_index=True)
        out_path = os.path.join(RESAMPLED_DIR, f"{sensor}_resampled.csv")
        resampled.to_csv(out_path, index=False)
        print(f"combined data saved to: {out_path}")
        print(f"final shape: {resampled.shape}")

    print("resampling done")
    return RESAMPLED_DIR

def segment(resampled_dir=RESAMPLED_DIR):
    print("\n[3] segmenting windows for all configs...")

    if MERGE_SENSORS:
        df = _build_merged_sensor_df(resampled_dir)
        if df.empty:
            print("merged IMU dataframe is empty, skip segmentation.")
            return WINDOW_DIR

        feat_cols = [
            c
            for c in df.columns
            if any(k in c for k in ["acc_", "gyro_", "magn_"])
        ]

        df = _zscore_per_user(df, feat_cols)

        for cfg in SEGMENT_CONFIGS:
            win_sec = cfg["win"]
            overlap = cfg["overlap"]

            win_size = int(win_sec * TARGET_FS)
            stride = int(win_size * (1 - overlap / 100))

            X, y, users = _generate_windows(df, feat_cols, win_size, stride)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            tag = f"IMU_win{win_sec}s_overlap{overlap}{_event_suffix()}"
            np.save(os.path.join(WINDOW_DIR, f"X_{tag}.npy"), X)
            np.save(os.path.join(WINDOW_DIR, f"y_{tag}.npy"), y)
            np.save(os.path.join(WINDOW_DIR, f"users_{tag}.npy"), users)

            print(f"{tag}: {X.shape[0]} windows saved.")

        print("segmentation done")
        return WINDOW_DIR

    for sensor in SENSORS:
        fpath = os.path.join(resampled_dir, f"{sensor}_resampled.csv")
        if not os.path.exists(fpath):
            print(f"{sensor}: resampled file not found, skip.")
            continue

        df = pd.read_csv(fpath)
        df = df[df["emotion"].isin(EMOTIONS)].copy()
        if df.empty:
            print(f"{sensor}: no matching emotion data after resampling.")
            continue

        feat_cols = [
            c
            for c in df.columns
            if any(k in c for k in ["acc_", "gyro_", "magn_"])
        ]

        df = _zscore_per_user(df, feat_cols)

        for cfg in SEGMENT_CONFIGS:
            win_sec = cfg["win"]
            overlap = cfg["overlap"]

            win_size = int(win_sec * TARGET_FS)
            stride = int(win_size * (1 - overlap / 100))

            X, y, users = _generate_windows(df, feat_cols, win_size, stride)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            tag = f"{sensor}_win{win_sec}s_overlap{overlap}{_event_suffix()}"

            np.save(os.path.join(WINDOW_DIR, f"X_{tag}.npy"), X)
            np.save(os.path.join(WINDOW_DIR, f"y_{tag}.npy"), y)
            np.save(os.path.join(WINDOW_DIR, f"users_{tag}.npy"), users)

            print(f"{tag}: {X.shape[0]} windows saved.")

    print("segmentation done")
    return WINDOW_DIR

def transform_windows(
    window_dir=WINDOW_DIR,
    out_dir=RP_DIR,
    method="RP",
    image_size=128,
    resize=False,
):
    if resize and image_size <= 0:
        print(f"[warn] invalid image_size={image_size}, fallback to 600")
        image_size = 600

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_img{image_size}" if resize and method != "FEAT" else ""

    transform_fn = METHODS.get(method)
    if transform_fn is None:
        raise ValueError(f"unknown transform method: {method}")

    print(f"\n=== TRANSFORM START - method={method}, resize={resize} ===")

    for cfg in SEGMENT_CONFIGS:
        win = cfg["win"]
        overlap = cfg["overlap"]

        base_tag = f"win{win}s_overlap{overlap}{_event_suffix()}"

        if MERGE_SENSORS:
            tag = f"IMU_{base_tag}"
            print(f"\n--- processing {tag} ---")

            x_path = os.path.join(window_dir, f"X_{tag}.npy")
            y_path = os.path.join(window_dir, f"y_{tag}.npy")
            u_path = os.path.join(window_dir, f"users_{tag}.npy")

            if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(u_path)):
                print(f"[SKIP] missing window files for {tag}")
                continue

            X = np.load(x_path)
            y = np.load(y_path, allow_pickle=True)
            users = np.load(u_path, allow_pickle=True)

            print(f"loaded windows: {X.shape}")

            transformed = []
            for w in tqdm(X, desc=f"{method} IMU"):
                img = transform_fn(w, image_size=image_size, resize_flag=resize)
                transformed.append(img.astype(np.float32))

            X_out = np.stack(transformed)

            target_dir = FEATURE_DIR if method == "FEAT" else out_dir
            out_x = os.path.join(target_dir, f"X_{method}_{tag}{suffix}.npy")
            out_y = os.path.join(target_dir, f"y_{method}_{tag}{suffix}.npy")
            out_u = os.path.join(target_dir, f"users_{method}_{tag}{suffix}.npy")

            np.save(out_x, X_out)
            np.save(out_y, y)
            np.save(out_u, users)

            print(f"[DONE] saved {X_out.shape} -> {out_x}")
            continue

        for sensor in SENSORS:

            tag = f"{sensor}_{base_tag}"
            print(f"\n--- processing {tag} ---")

            x_path = os.path.join(window_dir, f"X_{tag}.npy")
            y_path = os.path.join(window_dir, f"y_{tag}.npy")
            u_path = os.path.join(window_dir, f"users_{tag}.npy")

            if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(u_path)):
                print(f"[SKIP] missing window files for {tag}")
                continue

            X = np.load(x_path)
            y = np.load(y_path, allow_pickle=True)
            users = np.load(u_path, allow_pickle=True)

            print(f"loaded windows: {X.shape}")

            transformed = []
            for w in tqdm(X, desc=f"{method} {sensor}"):
                img = transform_fn(w, image_size=image_size, resize_flag=resize)
                transformed.append(img.astype(np.float32))

            X_out = np.stack(transformed)

            target_dir = FEATURE_DIR if method == "FEAT" else out_dir
            out_x = os.path.join(target_dir, f"X_{method}_{tag}{suffix}.npy")
            out_y = os.path.join(target_dir, f"y_{method}_{tag}{suffix}.npy")
            out_u = os.path.join(target_dir, f"users_{method}_{tag}{suffix}.npy")

            np.save(out_x, X_out)
            np.save(out_y, y)
            np.save(out_u, users)

            print(f"[DONE] saved {X_out.shape} → {out_x}")

    print("\n=== TRANSFORM DONE ===")

def parse_args():
    parser = argparse.ArgumentParser(description="data preprocessing pipeline")

    parser.add_argument(
        "--run",
        type=str,
        default="transform",
        choices=["raw", "resample", "segment", "transform", "everything"],
        help="which stage to run"
    )

    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[6],
        help="window durations"
    )

    parser.add_argument(
        "--overlaps",
        type=int,
        nargs="+",
        default=[0],
        help="overlap percentages"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="RP",
        choices=["RP", "GAF", "SPEC", "MTF", "LINE", "FEAT"],
        help="transform method"
    )

    parser.add_argument(
        "--resize",
        type=str,
        default="False",
        choices=["True", "False"],
        help="resize images to fixed image_size"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="output image size (H=W)"
    )

    parser.add_argument(
        "--sensors",
        type=str,
        nargs="+",
        default=["Chest"],
        help="which sensors to include"
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=DEFAULT_EVENTS,
        help="which emotion events to include (e.g. E01 E02 E03 E04)"
    )
    parser.add_argument(
        "--merge-sensors",
        type=str,
        default="True",
        choices=["True", "False"],
        help="merge selected sensors (Chest/arms) into one dataset"
    )

    return parser.parse_args()


def update_globals_from_args(args):
    global SENSORS, SEGMENT_CONFIGS, MERGE_SENSORS, ACTIVE_EVENTS, EMOTIONS

    SENSORS = args.sensors
    MERGE_SENSORS = args.merge_sensors.lower() == "true"
    ACTIVE_EVENTS = list(args.events)
    EMOTIONS = [EMOTION_MAP[e] for e in ACTIVE_EVENTS if e in EMOTION_MAP]

    new_cfg = []
    for d in args.durations:
        for o in args.overlaps:
            new_cfg.append({"win": d, "overlap": o})

    SEGMENT_CONFIGS = new_cfg

    print("\nUPDATED CONFIG:")
    print("SENSORS:", SENSORS)
    print("MERGE_SENSORS:", MERGE_SENSORS)
    print("EVENTS:", ACTIVE_EVENTS)
    print("EMOTIONS:", EMOTIONS)
    print("SEGMENT_CONFIGS:", SEGMENT_CONFIGS)


def main():
    args = parse_args()
    update_globals_from_args(args)

    resize_flag = args.resize.lower() == "true"

    if args.run == "raw":
        process_sensor_data()

    elif args.run == "resample":
        resample_sensors()

    elif args.run == "segment":
        segment()

    elif args.run == "transform":
        transform_windows(
            window_dir=WINDOW_DIR,
            out_dir=RP_DIR,
            method=args.method,
            image_size=args.image_size,
            resize=resize_flag,
        )

    elif args.run == "everything":
        process_sensor_data()
        resample_sensors()
        segment()
        transform_windows(
            window_dir=WINDOW_DIR,
            out_dir=RP_DIR,
            method=args.method,
            image_size=args.image_size,
            resize=resize_flag,
        )


if __name__ == "__main__":
    main()
