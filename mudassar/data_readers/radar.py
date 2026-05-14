import os
import pickle
from colorsys import hsv_to_rgb
from glob import glob

import numpy as np

class RadarData:
    # "RF-Behavior/Radar/C1/U03/M08/01/" # "swipe-left"       # suggests that +y = left
    # "RF-Behavior/Radar/C1/U01/M15/03/" # "left-arm-circle"  # suggests that +y = left
    # "RF-Behavior/Radar/C1/U05/M07/01/" # "swipe-right"      # suggests that -y = right
    # "RF-Behavior/Radar/C1/U03/M16/01/" # "right-arm-circle" # suggests that -y = right
    # "RF-Behavior/Radar/C1/U01/M06/01/" # "lateral-to-front" # suggests that +x = front
    # "RF-Behavior/Radar/C1/U03/M11/01/" # "two-hand-throw"   # suggests that +x = front
    # +z is above

    def __init__(self, points, radar_ids, timestamps):
        self.points = points
        self.radar_ids = radar_ids
        self.timestamps = np.array(timestamps)
        self.path = None
        self.start = None
        self.fps = None

    @classmethod
    def read_from_path(cls, path:str, transform=True, normalize=False):
        pickles = cls.read_pickles(path)
        points, rids, timestamps = [], [], []
        for r_id, pkl in pickles.items():
            for frame in pkl:
                if len(frame):
                    coordinates = frame[:, 1:]
                    if transform:
                        coordinates = cls.transform(coordinates, r_id)
                        if normalize:
                            coordinates = cls.normalize(coordinates)
                    points.append(coordinates)
                    rids.append([r_id for _ in range(len(frame))])
                    timestamps.append(frame[0,0])
        rd = RadarData(points, rids, timestamps)
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
                pickles[int(os.path.basename(file).split('.')[0])] = pickle.load(f)
        return pickles

    def bin(self, fps=10, int_time=False):
        if fps is None or fps <=0:
            fps = self.mean_fps(self.timestamps)
        start, bins = self.make_time_bins(self.timestamps, fps=fps)
        points = [[] for _ in range(bins.max()+1)]
        radars = [[] for _ in range(bins.max()+1)]
        for p, r, b in zip(self.points, self.radar_ids, bins):
            points[b].extend(p.tolist())
            radars[b].extend(r)

        new_timestamps = [i for i, p in enumerate(points) if p]
        points = [np.array(points[i]) for i in new_timestamps]
        radars = [radars[i] for i in new_timestamps]
        if not int_time:
            new_timestamps = start + np.array(new_timestamps) / fps
            start, fps = None, None
        rd = RadarData(points, radars, new_timestamps)
        rd.start = start
        rd.fps = fps
        rd.path = self.path
        return rd

    @classmethod
    def make_time_bins(cls, timestamps, fps=10):
        if fps is None or fps <=0:
            fps = cls.mean_fps(timestamps)
        start = min(timestamps)
        return start, ((np.array(timestamps) - start) * fps).astype(int)

    @staticmethod
    def mean_fps(timestamps):
        return 1/ np.diff(np.sort(timestamps)).mean()

    @classmethod
    def normalize(cls, points):
         return (points - cls.MEANS) / cls.STDS

    MEANS = np.array([ 0.032089453582,  0.265011873078,  1.821485130146, 17.835674109973])
    STDS = np.array([1.21536512212 , 2.157523648347, 1.260789705373, 6.055741740021])

    @classmethod
    def transform(cls, points: np.ndarray, radar_id:int):
        if radar_id <= 4:
            transformed = cls.ceiling_transform(points, cls.CEILING_TRANSLATIONS[radar_id])
        else:
            transformed = cls.ground_transform_factory(cls.GROUND_THETA_DEGS[radar_id - 5])(points, cls.GROUND_TRANSLATIONS[radar_id - 5])
        return np.concatenate([transformed, points[:, -1:]], axis=-1)

    CEILING_TRANSLATIONS = [
        np.array([-2,  4, 5]), np.array([2,  4, 5]), np.array([0, 0, 5]),
        np.array([-2, -4, 5]), np.array([2, -4, 5])
    ]
    GROUND_THETA_DEGS = [180, 135, 90, 45, 0, -45, -90, -135]
    GROUND_TRANSLATIONS = [
        np.array([1.5, 0, 1.3]), np.array([1.06, -1.06, 1.3]),
        np.array([0, -1.5, 1.3]), np.array([-1.06, -1.06, 1.3]),
        np.array([-1.5, 0, 1.3]), np.array([-1.06, 1.06, 1.3]),
        np.array([0, 1.5, 1.3]), np.array([1.06, 1.06, 1.3])
    ]
    COLORS = [hsv_to_rgb(i/13, 1, 1) for i in range(13)]

    # --- Specific Math Logic for Ground ---
    @staticmethod
    def ground_transform_factory(deg):
        """Returns a transformation function locked to a specific angle."""
        theta_rad = np.deg2rad(deg)
        # Pre-calculate R matrix
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 1]
        ])

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
        y =  points[:, 1]
        x =  points[:, 2]

        return np.stack([x, y, z], axis=1) + translation

if __name__ == "__main__":
    path = "./RF-Behavior/Radar/C2/U01/A02/01/"
    rd = RadarData.read_from_path(path)
    rdb = rd.bin(fps=10, int_time=True)
    print(len(rdb.points), len(rdb.radar_ids), len(rdb.timestamps),rdb.timestamps, rdb.fps)