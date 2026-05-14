import csv
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

EMOTION_MAP = {
    "E01": "focus", "E02": "distraction", "E03": "stress",
    "E04": "relaxation", "E05": "depression", "E06": "excitement"
}

def load_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=4, low_memory=False)
    to_drop = [c for c in ["Frame", "Time (Seconds)", "TimeCode"] if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    df = df[pd.to_numeric(df[df.columns[0]], errors="coerce").notna()].reset_index(drop=True)
    return df


def parse_header(file_path: str) -> dict:
    """
    parses the first rows of a CSV and returns metadata for each column:
      { colname: { "name": <Chest|RightArm|...>,
                   "id": <sensor id>,
                   "type": <Rotation|Position>,
                   "coord": <X|Y|Z|W>,
                   "axis": "<type>_<coord>" } }
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()[:10]]

    reader = csv.reader(lines)
    rows = list(reader)

    names_row = next(r for r in rows if "Name" in r)
    ids_row = next(r for r in rows if "ID" in r)
    types_row = next(r for r in rows if ("Rotation" in r or "Position" in r))
    axes_row = next(r for r in rows if ("Frame" in r and "Time (Seconds)" in r and "TimeCode" in r))

    names = names_row[3:]
    ids_ = ids_row[3:]
    types = types_row[3:]
    axes = axes_row[3:]

    meta = {}
    counts = {}

    for name, sid, type_, coord in zip(names, ids_, types, axes):
        if not sid or sid == "ID" or name == "Name":
            continue

        name = name.strip()
        sid = sid.strip()
        type_ = type_.strip()
        coord = coord.strip()
        axis_full = f"{type_}_{coord}"

        idx = counts.get(sid, 0)
        colname = sid if idx == 0 else f"{sid}.{idx}"
        counts[sid] = idx + 1

        meta[colname] = {
            "name": name,
            "id": sid,
            "type": type_,
            "coord": coord,
            "axis": axis_full,
        }

    return meta


def parse_filename(filename: str):
    base = filename.replace(".csv", "")
    parts = base.split("_")
    user_id, emotion_id = parts[0], parts[2]
    return user_id, emotion_id, EMOTION_MAP.get(emotion_id, "unknown")


def normalize_wide(df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    df = df.copy()
    meta_cols = ["frame", "user_id", "emotion_id", "emotion"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("unknown method, use zscore or minmax")

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df


def normalize_long(df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    df = df.copy()
    if method == "zscore":
        grouped = df.groupby("axis")["value"].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    elif method == "minmax":
        grouped = df.groupby("axis")["value"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        raise ValueError("unknown method, use zscore or minmax")

    df["value"] = grouped
    return df


def _zscore_per_user(df: pd.DataFrame, feat_cols):
    def _z(g):
        return (g - g.mean()) / (g.std(ddof=0) + 1e-8)

    df = df.copy()
    df[feat_cols] = df.groupby("user_id", group_keys=False)[feat_cols].apply(_z)
    return df


def _generate_windows(df: pd.DataFrame, feat_cols, win_size, stride):
    X, y, users = [], [], []

    df = df.sort_values(["user_id", "emotion", "frame"]).reset_index(drop=True)

    for uid in df["user_id"].unique():
        df_u = df[df["user_id"] == uid]
        for emo in df_u["emotion"].unique():
            df_e = df_u[df_u["emotion"] == emo].reset_index(drop=True)
            L = len(df_e)
            if L < win_size:
                continue

            for start in range(0, L - win_size + 1, stride):
                end = start + win_size
                w = df_e.iloc[start:end]

                X.append(w[feat_cols].values.astype(np.float32))

                y.append(w["emotion"].iloc[len(w) // 2])

                users.append(uid)

    if not X:
        return np.empty((0, 0, 0), dtype=np.float32), np.array([]), np.array([])

    return np.stack(X), np.array(y), np.array(users)

def process_single_file_wide(file_path: str) -> pd.DataFrame:
    filename = os.path.basename(file_path)
    meta = parse_header(file_path)
    df = load_clean_data(file_path)

    target_parts = {"Chest", "RightArm", "LeftArm"}
    cols = [c for c in df.columns if c in meta and meta[c]["name"] in target_parts]
    if not cols:
        print(f"no target columns found in {filename}")
        return pd.DataFrame(columns=["user_id","emotion_id","emotion","frame","sensor","type","coord","axis","value"])

    df_sensors = df[cols].copy()
    df_sensors = df_sensors.reset_index().rename(columns={"index": "frame"})

    rename_map = {c: f"{meta[c]['name']}_{meta[c]['type']}_{meta[c]['coord']}" for c in cols}
    df_sensors = df_sensors.rename(columns=rename_map)

    user_id, emotion_id, emotion = parse_filename(filename)
    df_sensors["user_id"] = user_id
    df_sensors["emotion_id"] = emotion_id
    df_sensors["emotion"] = emotion

    meta_cols = ["frame", "user_id", "emotion_id", "emotion"]
    feature_cols = [c for c in df_sensors.columns if c not in meta_cols]
    df_sensors[feature_cols] = df_sensors[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    df_sensors = df_sensors[meta_cols + feature_cols]

    return df_sensors

def process_single_file_long(file_path: str) -> pd.DataFrame:
    filename = os.path.basename(file_path)
    meta = parse_header(file_path)
    df = load_clean_data(file_path)

    target_parts = {"Chest", "RightArm", "LeftArm"}
    cols = [c for c in df.columns if c in meta and meta[c]["name"] in target_parts]
    if not cols:
        print(f"no target columns found in {filename}")
        return pd.DataFrame(columns=["user_id","emotion_id","emotion","frame","sensor","type","coord","axis","value"])

    df_sensors = df[cols].copy()
    df_sensors["frame"] = df_sensors.index

    long = df_sensors.melt(id_vars="frame", var_name="colname", value_name="value")

    long["sensor"] = long["colname"].map(lambda c: meta[c]["name"])
    long["type"] = long["colname"].map(lambda c: meta[c]["type"])
    long["coord"] = long["colname"].map(lambda c: meta[c]["coord"])
    long["axis"] = long["colname"].map(lambda c: meta[c]["axis"])

    user_id, emotion_id, emotion = parse_filename(filename)
    long["user_id"] = user_id
    long["emotion_id"] = emotion_id
    long["emotion"] = emotion

    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])

    return long[["user_id","emotion_id","emotion","frame","sensor","type","coord","axis","value"]]


def _radar_extract_points(frame, use_density: bool = True) -> np.ndarray:
    pts = []
    for pt in frame:
        if len(pt) < 4:
            continue
        x, y, z = pt[1], pt[2], pt[3]
        if use_density and len(pt) >= 5:
            d = pt[4]
            pts.append([x, y, z, d])
        else:
            pts.append([x, y, z])
    if not pts:
        return np.empty((0, 4 if use_density else 3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def _radar_sample_points(pts: np.ndarray, points_per_frame: int, mode: str = "random") -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((points_per_frame, pts.shape[1]), dtype=np.float32)
    if mode == "topk_density" and pts.shape[1] >= 4:
        idx = np.argsort(pts[:, -1])[::-1]
        sel = pts[idx][:points_per_frame]
        if len(sel) < points_per_frame:
            pad = np.zeros((points_per_frame - len(sel), pts.shape[1]), dtype=np.float32)
            sel = np.vstack([sel, pad])
        return sel
    choice = np.random.choice(len(pts), points_per_frame, replace=True)
    return pts[choice]


def radar_time_bin_sequence(
    raw_seq,
    target_fps: float = 10.0,
    points_per_frame: int = 16,
    use_density: bool = True,
    sample_mode: str = "random",
):
    frames = []
    for frame in raw_seq:
        if len(frame) == 0:
            continue
        ts = float(frame[0][0])
        pts = _radar_extract_points(frame, use_density=use_density)
        if len(pts) == 0:
            continue
        frames.append((ts, pts))

    if not frames:
        return None

    frames.sort(key=lambda x: x[0])
    t0, t1 = frames[0][0], frames[-1][0]
    step = 1.0 / float(target_fps)
    bins = np.arange(t0, t1 + step, step)
    n_bins = max(len(bins) - 1, 1)

    dim = 4 if use_density else 3
    out = np.zeros((n_bins, points_per_frame, dim), dtype=np.float32)

    bin_points = [[] for _ in range(n_bins)]
    for ts, pts in frames:
        idx = int((ts - t0) // step)
        if idx < 0:
            continue
        if idx >= n_bins:
            idx = n_bins - 1
        bin_points[idx].append(pts)

    for i, plist in enumerate(bin_points):
        if not plist:
            continue
        pts = np.concatenate(plist, axis=0)
        if len(pts) == 0:
            continue
        out[i] = _radar_sample_points(pts, points_per_frame, mode=sample_mode)

    return out
