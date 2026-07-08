import os
from copy import deepcopy
from glob import glob
from typing import Literal
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch


def make_radar_path(dataset_dir, campaign, user, behavior, repetition):
    path = os.path.abspath(os.path.join(dataset_dir, "Radar",campaign,user,behavior, str(repetition)))
    if not os.path.exists(path):
        path = os.path.abspath("/0".join(os.path.split(path)))
    test_path = os.path.abspath("/0".join(os.path.split(path)))
    if os.path.exists(test_path):
        path = test_path
    if not os.path.exists(path):
        path = os.path.abspath(path.rstrip("/\\")+"-1")
    if not os.path.exists(path):
        path = os.path.abspath(path.rsplit("-", 1)[0]+"-2")
    if os.path.exists(path):
        return [os.path.normpath(p) for p in glob(f"{path}/*.pkl")]
    # print(f"Radar path not found for {campaign}/{user}/{behavior}/{repetition} at {path}")
    return None

def make_infrared_path(dataset_dir, campaign, user, behavior, repetition):
    path = os.path.abspath(os.path.join(dataset_dir, "InfraredCam", "InfraredCam", campaign, user, behavior, f"{repetition}.csv"))
    if os.path.exists(path):
        return os.path.normpath(path)
    # print(f"Infrared path not found for {campaign}/{user}/{behavior}/{repetition} at {path}")
    return None

def find_available_files(dataset_dir):
    def extract_id(path: str):
        parts = os.path.normpath(path).split(os.sep)[-4:]
        return parts[:3] + [int(parts[3].rsplit(".", 1)[0].rsplit("-", 1)[0])]

    paths = glob(os.path.join(dataset_dir, "InfraredCam", "InfraredCam", "C*", "U*", "[MAE]*", "*.csv"))
    c, u, b, r = zip(*[extract_id(p) for p in paths])
    df_ir = pd.DataFrame({"campaign": c, "user": u, "behavior": b, "repetition": r, "infrared_path": paths})

    paths = glob(os.path.join(dataset_dir, "Radar", "C*", "U*", "[MAE]*", "*[0-9]"))
    c, u, b, r = zip(*[extract_id(p) for p in paths])
    radar_paths = [glob(os.path.join(p, "*.pkl")) for p in paths]
    df_r = pd.DataFrame({"campaign": c, "user": u, "behavior": b, "repetition": r, "radar_path": radar_paths})

    key_cols = ["campaign", "behavior", "user", "repetition"]
    merged = pd.merge(df_ir, df_r, how="outer", on=key_cols)
    return merged.sort_values(by=key_cols).reset_index(drop=True)

def get_common_users(df: pd.DataFrame):
    user_counts = df.groupby(["behavior", "user"]).count().repetition.reset_index()
    mask = user_counts.behavior.str.startswith("E")
    count_mask = user_counts.repetition == 8
    return sorted(set.intersection(*(user_counts[mask | (~mask & count_mask)].groupby("behavior").agg({"user":set})["user"].values.tolist())))


class BehaviorDataset(torch.utils.data.Dataset):
    REPEATED_ACTIVITIES = ("A01", "A02")
    REPEATED_EMOTIONS = ("E01", "E02", "E03", "E04", "E05", "E06")
    def __init__(
        self,
        dataframe: pd.DataFrame,
        modality: Literal["radar", "infrared"],
        augment_rate: float = 0.0,
        radar_bin_fps: float = 18.7,
    ):
        if "behavior" not in dataframe.columns:
            raise ValueError("Dataframe must contain a 'behavior' column")
        self.df = dataframe.reset_index(drop=True)
        self.modality = modality
        self.augment_rate = augment_rate
        self.label_names = sorted(self.df["behavior"].unique().tolist())
        self.users = sorted(self.df["user"].unique().tolist()) if "user" in self.df.columns else None
        self.class_weights = self._make_class_weights(self.df, self.label_names)

        self._cache = {}
        self.labels = self.df["behavior"].apply(self.label_names.index).tolist()

        self.path_column = f"{modality}_path"
        if modality == "radar":
            from .radar import RadarData
            self.reader_cls = RadarData
            self.radar_bin_fps = radar_bin_fps
        elif modality == "infrared":
            from .infrared import InfraredData
            self.reader_cls = InfraredData
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self.repeated_activities = deepcopy(self.REPEATED_ACTIVITIES)
        self.repeated_emotions = deepcopy(self.REPEATED_EMOTIONS)

    def __len__(self):
        # if self.users and len(self.df) < (expected_dataset_size := len(self.users) * len(self.label_names) * 8):
        #     return expected_dataset_size
        return len(self.df)

    def _load_sample(self, idx: int):
        if idx not in self._cache:
            path = self.df.iloc[idx][self.path_column]
            sample = self.reader_cls.read_from_path(path, transform=True, normalize=True, dtype=np.float32)
            if self.modality == "radar":
                sample = sample.bin(self.radar_bin_fps)
                sample.frames = [torch.from_numpy(f).float().unsqueeze(0).unsqueeze(0) for f in sample.frames] # list[4D tensors]
            elif self.modality == "infrared":
                sample.frames = torch.from_numpy(sample.frames).float().unsqueeze(0)  # 4D tensor
                if self.df.iloc[idx]["behavior"].startswith("M"):
                    sample.frames = sample.frames[:, :, :3, :]
            self._cache[idx] = sample
        return self._cache[idx]

    def __getitem__(self, idx: int):
        idx = idx % len(self.df)
        sample = self._load_sample(idx)
        data = sample.augmented_data() if np.random.rand() < self.augment_rate else sample.raw_data()
        if self.df.iloc[idx]["behavior"] in self.repeated_activities + self.repeated_emotions:
            # multiplier = 3 if self.df.iloc[idx]["behavior"] in self.repeated_emotions else 1
            actual_frame_count = data.shape[1] if isinstance(data, torch.Tensor) else len(data)
            selected_frame_count = round(min(np.random.uniform(4,6)/(sample.timestamps[-1] - sample.timestamps[0]), 1.0) * actual_frame_count)
            start = np.random.randint(0, actual_frame_count - selected_frame_count + 1)
            data = data[:, start:start + selected_frame_count] if isinstance(data, torch.Tensor) else data[start:start + selected_frame_count]
            # print(f"Selected frame count: {selected_frame_count} / Actual frame count: {actual_frame_count} / raw frame count: {sample.n_frames} / Start: {start} / Sample: {self.df.iloc[idx]['behavior']} / shape: {data.shape if isinstance(data, torch.Tensor) else len(data)}")
        label = self.labels[idx]
        return data, label

    def _make_class_weights(self, df, labels, dtype=torch.float32):
        counts = torch.from_numpy(df["behavior"].value_counts()[labels].to_numpy())
        weights = counts.sum() / (len(counts) * counts)
        return weights.to(dtype=dtype)

    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        return list(inputs), torch.tensor(labels, dtype=torch.long)

    def preload(self, verbose=True, num_workers=8, desc="Preloading dataset"):
        rng = range(len(self.df))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            m = executor.map(self._load_sample, rng)
            if verbose:
                from tqdm.auto import tqdm
                m = tqdm(m, total=len(self), desc=desc, unit="sample")
            list(m)
        return self
