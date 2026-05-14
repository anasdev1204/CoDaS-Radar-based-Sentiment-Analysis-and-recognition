import os
from colorsys import hsv_to_rgb

import numpy as np
import pandas as pd


class InfraredData:
    # +y is above
    def __init__(self, landmarks, timestamps, path=None):
        self.landmarks = landmarks
        self.timestamps = timestamps
        self.path = path

    @classmethod
    def read_from_path(cls, path: str, transform=True, normalize=False):
        df = cls.read_csv(path)

        timestamps = df[cls.TIMESTAMP_COLUMN].astype(np.float32).to_numpy()
        landmarks = df.filter(items=cls.COLUMNS).astype(np.float32).to_numpy().reshape(len(df), -1, 7)

        # bug: swap LeftArm and RightArm (shouldn't really affect the model but is confusing for visualization)
        landmarks[:, [1, 2]] = landmarks[:, [2, 1]]

        if transform:
            landmarks = cls.transform(landmarks)
            if normalize:
                landmarks = cls.normalize(landmarks)
        return cls(landmarks, timestamps, path=path)

    @classmethod
    def read_csv(cls, path: str):
        if not os.path.isfile(path):
            raise ValueError(f"No file found at '{path}'")

        df1 = pd.read_csv(path, skiprows=[0,1,2,4], nrows=3, header=None)
        df2 = pd.read_csv(path, skiprows=7, header=None)
        columns = df1.iloc[0].fillna("") + "_" + df1.iloc[1].fillna("") + "_" + df1.iloc[2].fillna("")
        df = df2.rename(columns=columns)

        df = df.filter(items=[cls.TIMESTAMP_COLUMN] + cls.COLUMNS).dropna()

        return df
        # campaign = os.path.normpath(path).split(os.sep)[-4]
        # return cls.fix_column_names(df, campaign)

    @staticmethod
    def transform(landmarks: np.ndarray):
        # todo: align axes with radar data
        return landmarks

    @classmethod
    def normalize(cls, landmarks: np.ndarray):
        return (landmarks - cls.MEANS) / cls.STDS

    MEANS = np.array([[[3.248380930763, 922.046612473506, -240.151300320984, -0.001156634352, 0.00849719666 , 0.000139222646, -0.00817210628]]])
    STDS = np.array([[[473.229016340735, 193.943481061356, 218.677452467549, 0.484396761754, 0.267481151044, 0.249025394452, 0.588057615433]]])

    # @staticmethod
    # def fix_column_names(df:pd.DataFrame, campaign:str):
    #     mapper = {f"{a}_{pt}_{coord}": f"{b}_{pt}_{coord}" for a,b in [("LeftArm", "RightArm"), ("RightArm", "LeftArm")] for pt, coords in [("Position", "XYZ"), ("Rotation", "XYZW")]for coord in coords}
    #     return df.rename(columns=mapper)

    BODY_PARTS = ["Chest", "LeftArm", "RightArm", "Hips", "LeftLeg", "RightLeg"]
    COLUMNS = [
        f"{bp}_{pt}_{coord}"
        for bp in BODY_PARTS
        for pt, coords in [("Position", "XYZ"), ("Rotation", "XYZW")]
        for coord in coords
    ]
    TIMESTAMP_COLUMN = "Name__Time (Seconds)"

