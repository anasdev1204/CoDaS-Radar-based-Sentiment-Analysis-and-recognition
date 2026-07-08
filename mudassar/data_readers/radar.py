import os
import pickle
from colorsys import hsv_to_rgb
from copy import deepcopy
from glob import glob

import numpy as np

class RadarData:
    # +z is above

    # "RF-Behavior/Radar/C1/U03/M08/01/" # "swipe-left"       # suggests that +y = left
    # "RF-Behavior/Radar/C1/U01/M15/03/" # "left-arm-circle"  # suggests that +y = left
    # "RF-Behavior/Radar/C1/U05/M07/01/" # "swipe-right"      # suggests that -y = right
    # "RF-Behavior/Radar/C1/U03/M16/01/" # "right-arm-circle" # suggests that -y = right
    # "RF-Behavior/Radar/C1/U01/M06/01/" # "lateral-to-front" # suggests that +x = front
    # "RF-Behavior/Radar/C1/U03/M11/01/" # "two-hand-throw"   # suggests that +x = front

    def __init__(self, frames, radar_ids, timestamps, unique_rids=None):
        self.frames = frames
        self.radar_ids = radar_ids
        self.timestamps = np.array(timestamps)
        self.path = None
        self.start = None
        self.fps = None
        if unique_rids is None:
            self.unique_rids = np.array(sorted({i for rids in radar_ids for i in rids}))
        else:
            self.unique_rids = unique_rids

    @classmethod
    def read_from_path(cls, path: str, transform=True, normalize=False, dtype=None):
        pickles = cls.read_pickles(path)
        points, rids, timestamps, unique_rids = [], [], [], set()
        for r_id, pkl in pickles.items():
            for frame in pkl:
                if len(frame):
                    coordinates = frame[:, 1:]
                    if transform:
                        coordinates = cls.transform(coordinates, r_id)
                        if normalize:
                            coordinates = cls.normalize(coordinates)
                    if dtype is not None:
                        coordinates = coordinates.astype(dtype)
                    points.append(coordinates)
                    rids.append([r_id for _ in range(len(frame))])
                    timestamps.append(frame[0, 0])
            unique_rids.add(r_id)
        rd = RadarData(points, rids, timestamps, unique_rids=np.array(sorted(unique_rids)))
        rd.path = path
        return rd

    @classmethod
    def read_pickles(cls, path):
        pickles = {}
        if isinstance(path, list):
            files = path
        else:
            if os.path.isdir(path):
                path = os.path.join(path, "*.pkl")
            files = glob(path)
        if not files:
            raise ValueError(f"No files found at '{path}'")
        for file in files:
            with open(file, "rb") as f:
                pickles[int(os.path.basename(file).split(".")[0])] = pickle.load(f)
        return pickles

    def bin(self, fps=10.0, int_time=False):
        if fps is None or fps <= 0:
            fps = self.mean_fps(self.timestamps)
        start, bins = self.make_time_bins(self.timestamps, fps=fps)
        points = [[] for _ in range(bins.max() + 1)]
        radars = [[] for _ in range(bins.max() + 1)]
        for p, r, b in zip(self.frames, self.radar_ids, bins):
            points[b].extend(p.tolist())
            radars[b].extend(r)

        new_timestamps = [i for i, p in enumerate(points) if p]
        points = [np.array(points[i]) for i in new_timestamps]
        radars = [radars[i] for i in new_timestamps]
        if not int_time:
            # new_timestamps = start + np.array(new_timestamps) / fps
            # start, fps = None, None
            new_timestamps = np.array(new_timestamps) / fps
            fps = None
        rd = RadarData(points, radars, new_timestamps)
        rd.start = start
        rd.fps = fps
        rd.path = self.path
        return rd

    @classmethod
    def make_time_bins(cls, timestamps, fps=10.0):
        if fps is None or fps <= 0:
            fps = cls.mean_fps(timestamps)
        start = min(timestamps)
        return start, ((np.array(timestamps) - start) * fps).astype(int)

    def raw_data(self) -> list:
        return self.frames

    def augmented_data(self, low: float=0.7, high: float=1, by_rid=0.5):
        if np.random.rand() < by_rid:
            return self.augment_by_rid(low=low, high=high)
        else:
            return self.augment_by_points(low=low, high=high)

    def augment_by_rid(self, low: float=0.7, high: float=1) -> list:
        n = round(len(self.unique_rids) * np.random.uniform(low, high))
        idx = np.arange(len(self.unique_rids))
        np.random.shuffle(idx)
        keep_ids = self.unique_rids[idx[:n]]

        sample = [self._apply_mask(frame, np.isin(rids, keep_ids)) for frame, rids in zip(self.frames, self.radar_ids)]
        return sample

    def augment_by_points(self, low: float=0.7, high: float=1) -> list:
        lens = [f.shape[-2] for f in self.frames]
        min_len, max_len = min(lens), max(lens)
        len_gap = max_len - min_len
        frac_gap = high - low
        sample = []
        for f in self.frames:
            n_true = round(f.shape[-2] * np.random.uniform(high - ((f.shape[-2] - min_len)/len_gap * frac_gap), high))
            mask = np.zeros(f.shape[-2], dtype=bool)
            mask[np.random.choice(f.shape[-2], n_true, replace=False)] = True
            sample.append(self._apply_mask(f, mask))

        return sample

    def pad(self, min_density=None, max_points=None):
        if min_density is None:
            actual_max_points = max(f.shape[-2] for f in self.frames)
        else:
            actual_max_points = max((d[:, -1] > min_density).sum() for d in self.frames)
        max_points = max(max_points or 0, actual_max_points)

        points, r_ids = [], []
        for f, r in zip(self.frames, self.radar_ids):
            if min_density is None:
                pad_width = max_points - f.shape[-2]
            else:
                mask = f[:, -1] > min_density
                pad_width = max_points - mask.sum()
                f = f[mask]
                r = [r[i] for i in range(f.shape[-2]) if mask[i]]

            if f.shape[-2] > 0:
                points.append(f.tolist() + [(0, 0, 0, 0)] * pad_width)
                r_ids.append(r + [None] * pad_width)

        rd = deepcopy(self)
        rd.frames = points
        rd.radar_ids = r_ids
        return rd

    @property
    def n_frames(self):
        return len(self.frames)

    def animate(self, fps=10, **kwargs):
        rd = self
        if rd.start is None:
            rd = rd.bin(fps=fps)
        rd = rd.pad(min_density=kwargs.get("min_density"), max_points=kwargs.get("max_points"))
        points = np.array(rd.frames)[:, :, :3]
        print(points.shape)
        scatter_colors = [[(RadarData.COLORS[i] if i is not None else (0, 0, 0, 0)) for i in r] for r in rd.radar_ids]

        from .visualize import MatPlot3D
        import matplotlib.pyplot as plt
        anim = MatPlot3D.animate(points, scatter_colors=scatter_colors, **kwargs)
        plt.close()
        return anim

    def show(self, fps=10, vertical_axis="z", azimuth=10, elevation=20, **kwargs):
        from IPython.display import HTML, display

        display(HTML(self.animate(
            fps=fps, vertical_axis=vertical_axis, azimuth=azimuth, elevation=elevation, **kwargs
        ).to_jshtml(fps=fps)))

    @staticmethod
    def mean_fps(timestamps):
        return 1 / np.diff(np.sort(timestamps)).mean()

    @classmethod
    def normalize(cls, points):
        return (points - cls.MEANS) / cls.STDS

    MEANS = np.array([0.032089453582, 0.265011873078, 1.821485130146, 17.835674109973])
    STDS = np.array([1.21536512212, 2.157523648347, 1.260789705373, 6.055741740021])

    @classmethod
    def transform(cls, points: np.ndarray, radar_id: int):
        if radar_id <= 4:
            transformed = cls.ceiling_transform(points, cls.CEILING_TRANSLATIONS[radar_id])
        else:
            transformed = cls.ground_transform_factory(cls.GROUND_THETA_DEGS[radar_id - 5])(
                points, cls.GROUND_TRANSLATIONS[radar_id - 5]
            )
        return np.concatenate([transformed, points[:, -1:]], axis=-1)

    CEILING_TRANSLATIONS = [
        np.array([-2, 4, 5]),
        np.array([2, 4, 5]),
        np.array([0, 0, 5]),
        np.array([-2, -4, 5]),
        np.array([2, -4, 5]),
    ]
    GROUND_THETA_DEGS = [180, 135, 90, 45, 0, -45, -90, -135]
    GROUND_TRANSLATIONS = [
        np.array([1.5, 0, 1.3]),
        np.array([1.06, -1.06, 1.3]),
        np.array([0, -1.5, 1.3]),
        np.array([-1.06, -1.06, 1.3]),
        np.array([-1.5, 0, 1.3]),
        np.array([-1.06, 1.06, 1.3]),
        np.array([0, 1.5, 1.3]),
        np.array([1.06, 1.06, 1.3]),
    ]
    COLORS = [hsv_to_rgb(i / 13, 1, 1) for i in range(13)]

    # --- Specific Math Logic for Ground ---
    @staticmethod
    def ground_transform_factory(deg):
        """Returns a transformation function locked to a specific angle."""
        theta_rad = np.deg2rad(deg)
        # Pre-calculate R matrix
        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0], [np.sin(theta_rad), np.cos(theta_rad), 0], [0, 0, 1]])

        def transform(points, translation):
            # Apply rotation then translation
            # Note: points[:, 0] is X, points[:, 1] is Y, points[:, 2] is Z
            res = R @ [points[:, 0], points[:, 1], points[:, 2]]
            x, y, z = res[0], res[1], res[2]
            return np.stack([x, y, z], axis=1) + translation

        return transform

    @staticmethod
    def ceiling_transform(points, translation):
        # Logic from original script: z = -points[:,0], y = points[:,1], x = points[:,2]
        z = -points[:, 0]
        y = points[:, 1]
        x = points[:, 2]

        return np.stack([x, y, z], axis=1) + translation

    def _apply_mask(self, f, mask):
        try:
            import torch
            if isinstance(f, torch.Tensor):
                mask = torch.from_numpy(mask)
        except ImportError:
            pass
        return f[..., mask, :]

if __name__ == "__main__":
    path = "./RF-Behavior/Radar/C2/U01/A02/01/"
    rd_ = RadarData.read_from_path(path)
    rdb = rd_.bin(fps=10, int_time=True)
    print(len(rdb.frames), len(rdb.radar_ids), len(rdb.timestamps), rdb.timestamps, rdb.fps)
