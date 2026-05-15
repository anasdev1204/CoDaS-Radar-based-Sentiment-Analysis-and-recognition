import argparse
import os
import pickle
import re
from typing import Iterable, Iterator
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
from natsort import natsorted
from pyts.image import RecurrencePlot
from skimage.transform import resize
from tqdm import tqdm

from transforms import METHODS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "radar"))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "processed", "radar"))
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
NUM_RADARS = 13

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*numpy\\.core\\.numeric.*",
)

def _out_name(event_ids: list[str]) -> str:
    return f"radar_samples_{'_'.join(sorted(event_ids))}.npz"


def _event_suffix(event_ids: list[str]) -> str:
    if sorted(event_ids) == sorted(DEFAULT_EVENTS):
        return ""
    return "_" + "_".join(sorted(e.lower() for e in event_ids))


def _resize_img(img: np.ndarray, out_size: int) -> np.ndarray:
    return resize(img, (out_size, out_size), anti_aliasing=True).astype(np.float32)


def _normalize_label(label: str) -> str:
    if isinstance(label, (np.integer, int)):
        val = int(label)
        if 0 <= val <= 5:
            label = f"E{val + 1:02d}"
        elif 1 <= val <= 6:
            label = f"E{val:02d}"
        else:
            return str(label)
    if label in EMOTION_MAP:
        return EMOTION_MAP[label]
    if label in EMOTION_MAP.values():
        return label
    return label


def _iter_radar_files(raw_dir: str) -> Iterator[tuple[str, str, str, str, str]]:
    for user_folder in natsorted(os.listdir(raw_dir)):
        user_path = os.path.join(raw_dir, user_folder)
        if not os.path.isdir(user_path):
            continue

        for label_folder in natsorted(os.listdir(user_path)):
            label_path = os.path.join(user_path, label_folder)
            if not os.path.isdir(label_path):
                continue

            for trial_folder in natsorted(os.listdir(label_path)):
                trial_path = os.path.join(label_path, trial_folder)
                if not os.path.isdir(trial_path):
                    continue

                for radar_file in natsorted(os.listdir(trial_path)):
                    if radar_file.endswith(".pkl"):
                        yield user_folder, label_folder, trial_folder, radar_file, os.path.join(trial_path, radar_file)


def _parse_ids(user_folder: str, label_folder: str, trial_folder: str, radar_file: str) -> tuple[str, str, int, int]:
    u_id = int(re.search(r"\d+", user_folder).group())
    l_id = int(re.search(r"\d+", label_folder).group())
    t_id = int(re.search(r"\d+", trial_folder).group())
    r_match = re.search(r"\d+", radar_file)
    r_id = int(r_match.group()) if r_match else 0
    return f"U{u_id:02d}", f"E{l_id:02d}", t_id, r_id


def _extract_points_with_dnc(frame: Iterable, use_density: bool = True) -> np.ndarray:
    pts = []
    for pt in frame:
        if len(pt) < 4:
            continue
        x, y, z = float(pt[1]), float(pt[2]), float(pt[3])
        dnc = float(pt[4]) if (use_density and len(pt) >= 5) else 0.0
        pts.append([x, y, z, dnc])
    if not pts:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def select_topk_repeat(points: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape (N, 4)")

    n = points.shape[0]
    if n == 0:
        return np.zeros((k, 4), dtype=np.float32)

    order = np.argsort(-points[:, 3], kind="mergesort")
    sorted_points = points[order]

    if n >= k:
        return sorted_points[:k].astype(np.float32, copy=False)

    reps = int(np.ceil(k / n))
    tiled = np.tile(sorted_points, (reps, 1))
    return tiled[:k].astype(np.float32, copy=False)


def preprocess_radar_sequence_topk(
    raw_seq: Iterable,
    target_fps: float,
    k: int,
    use_density: bool = True,
) -> Optional[np.ndarray]:
    frames: list[tuple[float, np.ndarray]] = []
    for frame in raw_seq:
        if len(frame) == 0:
            continue
        ts = float(frame[0][0])
        pts = _extract_points_with_dnc(frame, use_density=use_density)
        if pts.shape[0] == 0:
            continue
        frames.append((ts, pts))

    if not frames:
        return None

    frames.sort(key=lambda x: x[0])
    t0, t1 = frames[0][0], frames[-1][0]
    step = 1.0 / float(target_fps)
    bins = np.arange(t0, t1 + step, step)
    n_bins = max(len(bins) - 1, 1)

    out = np.zeros((n_bins, k, 4), dtype=np.float32)
    bin_points: list[list[np.ndarray]] = [[] for _ in range(n_bins)]

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
        out[i] = select_topk_repeat(pts, k)

    return out


def _align_radar_lengths(radar_data_list: List[np.ndarray], mode: str) -> Optional[List[np.ndarray]]:
    lengths = [d.shape[0] for d in radar_data_list]
    if len(set(lengths)) == 1:
        return radar_data_list
    if mode == "strict":
        return None
    min_len = min(lengths)
    return [d[:min_len] for d in radar_data_list]


def _to_object_array(items: List[np.ndarray]) -> np.ndarray:
    """Create a 1D object array without triggering ragged broadcasting."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def process_radar_data(
    raw_dir: str,
    events: list[str],
    fps: float,
    points: int,
    use_density: bool,
    align: str,
    out_name: str,
) -> Union[Tuple[List[np.ndarray], np.ndarray, np.ndarray], Tuple[None, None, None]]:
    print("\n[1] loading raw radar data...")

    if not os.path.exists(raw_dir):
        print(f"raw dir not found: {raw_dir}")
        return None, None, None

    raw_data = []
    events_norm = [_normalize_label(e) for e in events]
    event_ids = [e for e in events if isinstance(e, str) and e.startswith("E")]
    event_emotions = [_normalize_label(e) for e in event_ids]
    allow = set(events_norm + event_emotions)

    for user_folder, label_folder, trial_folder, radar_file, file_path in _iter_radar_files(raw_dir):
        try:
            user_id, emotion_id, trial_id, radar_id = _parse_ids(user_folder, label_folder, trial_folder, radar_file)
            emotion = _normalize_label(emotion_id)
            if emotion not in allow and emotion_id not in allow:
                continue

            with open(file_path, "rb") as f:
                raw_seq = pickle.load(f, encoding="latin1")

            fixed_seq = preprocess_radar_sequence_topk(
                raw_seq=raw_seq,
                target_fps=fps,
                k=points,
                use_density=use_density,
            )
            if fixed_seq is None:
                continue

            raw_data.append(
                {
                    "data": fixed_seq,
                    "user": user_id,
                    "label": emotion,
                    "trial": trial_id,
                    "radar": radar_id,
                }
            )
        except Exception as e:
            print(f"error processing {file_path}: {e}")

    if not raw_data:
        print("no valid radar samples found")
        return None, None, None

    data_list: list[np.ndarray] = []
    labels_list: list[str] = []
    users_list: list[str] = []

    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for item in raw_data:
        key = (item["user"], item["label"], item["trial"])
        grouped.setdefault(key, []).append(item)

    for (u, l, _), group in grouped.items():
        group = sorted(group, key=lambda x: x["radar"])
        radar_data_list = [g["data"] for g in group]
        if len(radar_data_list) != NUM_RADARS:
            continue
        aligned = _align_radar_lengths(radar_data_list, mode=align)
        if aligned is None:
            continue
        sample = np.stack(aligned, axis=0).astype(np.float32)
        data_list.append(sample)
        labels_list.append(l)
        users_list.append(u)

    if not data_list:
        print("no trials with exactly 13 radars were found")
        return None, None, None

    labels = np.asarray(labels_list)
    users = np.asarray(users_list)

    out_path = os.path.join(PROCESSED_DIR, out_name)
    np.savez_compressed(
        out_path,
        data=_to_object_array(data_list),
        labels=labels,
        users=users,
        fps=np.asarray([fps], dtype=np.float32),
        points=np.asarray([points], dtype=np.int32),
        use_density=np.asarray([int(use_density)], dtype=np.int32),
    )

    print(f"saved samples to: {out_path}")
    print(f"final count: data {len(data_list)}, labels {labels.shape}, users {users.shape}")
    print("sample shape: (13, T, K, 4)")
    return data_list, labels, users


def build_windows(
    data: Union[np.ndarray, List[np.ndarray]],
    labels: np.ndarray,
    users: np.ndarray,
    win_seconds: int,
    overlap: int,
    fps: float,
    event_ids: list[str],
) -> Optional[str]:
    if data is None or (isinstance(data, np.ndarray) and data.size == 0):
        print("no data to build windows")
        return None

    samples = list(data) if not (isinstance(data, np.ndarray) and data.dtype != object) else [data[i] for i in range(data.shape[0])]

    win_frames = max(int(round(win_seconds * fps)), 1)
    stride = max(int(win_frames * (1 - overlap / 100)), 1)

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    u_list: list[str] = []

    for i, sample in enumerate(samples):
        if sample is None:
            continue
        _, frames, _, _ = sample.shape
        if win_frames > frames:
            continue

        for start in range(0, frames - win_frames + 1, stride):
            end = start + win_frames
            X_list.append(sample[:, start:end, :, :])
            y_list.append(labels[i])
            u_list.append(users[i])

    if X_list:
        X = np.stack(X_list).astype(np.float32)
    else:
        X = np.empty((0, NUM_RADARS, win_frames, 0, 4), dtype=np.float32)

    y = np.asarray(y_list)
    u = np.asarray(u_list)

    tag = f"win{win_seconds}s_overlap{overlap}{_event_suffix(event_ids)}"
    np.save(os.path.join(WINDOW_DIR, f"X_{tag}.npy"), X)
    np.save(os.path.join(WINDOW_DIR, f"y_{tag}.npy"), y)
    np.save(os.path.join(WINDOW_DIR, f"users_{tag}.npy"), u)

    print(f"windows saved: X {X.shape}, y {y.shape}, users {u.shape}")
    return tag


def _load_windows(tag: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    x_path = os.path.join(WINDOW_DIR, f"X_{tag}.npy")
    y_path = os.path.join(WINDOW_DIR, f"y_{tag}.npy")
    u_path = os.path.join(WINDOW_DIR, f"users_{tag}.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(u_path)):
        print(f"missing window files for tag: {tag}")
        return None, None, None
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path, allow_pickle=True)
    users = np.load(u_path, allow_pickle=True)
    return X, y, users


def _flatten_window_for_sequence_model(window: np.ndarray) -> np.ndarray:
    _, t_len, k_val, d_val = window.shape
    return window.transpose(1, 0, 2, 3).reshape(t_len, NUM_RADARS * k_val * d_val)


def _generate_rp_points(window: np.ndarray, image_size: int = 128, resize_flag: bool = True) -> np.ndarray:
    _, _, k_val, d_val = window.shape
    points_total = NUM_RADARS * k_val
    agg = window.mean(axis=1).reshape(points_total, d_val)

    rp = RecurrencePlot(threshold="point", percentage=20)
    images = []
    for d in range(d_val):
        sig = agg[:, d].reshape(1, -1)
        rp_img = rp.fit_transform(sig)[0]
        if resize_flag:
            rp_img = _resize_img(rp_img, image_size)
        images.append(rp_img.astype(np.float32))
    return np.stack(images, axis=0)


def transform_windows(tag: str, method: str = "RP", image_size: int = 128, resize: bool = False) -> None:
    if resize and image_size <= 0:
        print(f"[warn] invalid image_size={image_size}, fallback to 128")
        image_size = 128

    X, y, users = _load_windows(tag)
    if X is None:
        return

    transformed = []
    suffix = f"_img{image_size}" if resize and method != "FEAT" else ""
    for w in tqdm(X, desc=f"{method} {tag}"):
        if method == "RP":
            out = _generate_rp_points(w, image_size=image_size, resize_flag=resize)
        else:
            flat = _flatten_window_for_sequence_model(w)
            transform_fn = METHODS.get(method)
            if transform_fn is None:
                raise ValueError(f"unknown transform method: {method}")
            out = transform_fn(flat, image_size=image_size, resize_flag=resize)
        transformed.append(out.astype(np.float32))

    X_out = np.stack(transformed) if transformed else np.empty((0,))
    target_dir = FEATURE_DIR if method == "FEAT" else RP_DIR

    np.save(os.path.join(target_dir, f"X_{method}_{tag}{suffix}.npy"), X_out)
    np.save(os.path.join(target_dir, f"y_{method}_{tag}{suffix}.npy"), y)
    np.save(os.path.join(target_dir, f"users_{method}_{tag}{suffix}.npy"), users)

    print(f"saved: X {X_out.shape}, y {y.shape}, users {users.shape}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="radar preprocessing pipeline")

    parser.add_argument(
        "--run",
        type=str,
        default="everything",
        choices=["clean", "segment", "transform", "everything"],
        help="which stage to run",
    )
    parser.add_argument(
        "--events",
        type=str,
        nargs="+",
        default=DEFAULT_EVENTS,
        help="which emotion events to include (e.g., E03 E04)",
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[6],
        help="window durations (seconds)",
    )
    parser.add_argument(
        "--overlaps",
        type=int,
        nargs="+",
        default=[0],
        help="overlap percentages",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="fps used in time-binning (seconds -> frames)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=16,
        help="K points per frame per radar",
    )
    parser.add_argument(
        "--use_density",
        type=str,
        default="True",
        choices=["True", "False"],
        help="use dnc if available",
    )
    parser.add_argument(
        "--align",
        type=str,
        default="min",
        choices=["min", "strict"],
        help="alignment mode for 13 radars",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="RP",
        choices=["FEAT", "RP"],
        help="transform method",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="window tag for --run transform",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="False",
        choices=["True", "False"],
        help="resize output to image_size",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="output image size (H=W)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize_flag = args.resize.lower() == "true"
    use_density = args.use_density.lower() == "true"

    if args.run == "clean":
        process_radar_data(
            raw_dir=DATA_DIR,
            events=args.events,
            fps=args.fps,
            points=args.points,
            use_density=use_density,
            align=args.align,
            out_name=_out_name(args.events),
        )

    elif args.run == "segment":
        path = os.path.join(PROCESSED_DIR, _out_name(args.events))
        if not os.path.exists(path):
            print(f"missing processed file: {path}")
            return
        npz = np.load(path, allow_pickle=True)
        data, labels, users = npz["data"], npz["labels"], npz["users"]
        data = list(data) if isinstance(data, np.ndarray) and data.dtype == object else data
        for win_seconds in args.durations:
            for overlap in args.overlaps:
                build_windows(data, labels, users, win_seconds, overlap, args.fps, args.events)

    elif args.run == "transform":
        if not args.tag:
            print("missing --tag for transform")
            return
        transform_windows(args.tag, method=args.method, image_size=args.image_size, resize=resize_flag)

    elif args.run == "everything":
        data, labels, users = process_radar_data(
            raw_dir=DATA_DIR,
            events=args.events,
            fps=args.fps,
            points=args.points,
            use_density=use_density,
            align=args.align,
            out_name=_out_name(args.events),
        )
        if data is None:
            return
        for win_seconds in args.durations:
            for overlap in args.overlaps:
                tag = build_windows(data, labels, users, win_seconds, overlap, args.fps, args.events)
                if tag:
                    transform_windows(tag, method=args.method, image_size=args.image_size, resize=resize_flag)


if __name__ == "__main__":
    main()
