import os
from glob import glob

import pandas as pd


def make_radar_path(dataset_dir, campaign, user, behavior, repetition):
    path = os.path.abspath(os.path.join(dataset_dir, "Radar",campaign,user,behavior, str(repetition)))
    if not os.path.exists(path):
        path = os.path.abspath("/0".join(os.path.split(path)))
    if os.path.exists(path):
        return [os.path.normpath(p) for p in glob(f"{path}/*.pkl")]
    return None

def make_infrared_path(dataset_dir, campaign, user, behavior, repetition):
    path = os.path.abspath(os.path.join(dataset_dir, "InfraredCam", "InfraredCam", campaign, user, behavior, f"{repetition}.csv"))
    if os.path.exists(path):
        return os.path.normpath(path)
    return None

def find_available_files(dataset_dir):
    paths = glob(os.path.join(dataset_dir, "InfraredCam", "InfraredCam", "C*", "U*", "[MAE]*", "*.csv"))
    instances = [(p, os.path.normpath(p).split(os.sep)[-4:]) for p in paths]
    # instances = [i for i in instances if not (i[0] == "C3" and i[1] == "U41")] # has 3 infrared landmarks instead of 6 like all others in C3
    instances = sorted(tuple(instance[:3] + [int(instance[3].rsplit(".", 1)[0])])+(p,) for p, instance in instances)
    instances = [(*i, make_radar_path(dataset_dir, *i[:4])) for i in instances]
    c, u, b, r, irp, rdp = zip(*instances)

    return pd.DataFrame({"campaign": c, "user": u, "behavior": b, "repetition": r, "infrared_path": irp, "radar_path": rdp})

def get_common_users(df: pd.DataFrame):
    user_counts = df.groupby(["behavior", "user"]).count().repetition
    return sorted(set.intersection(*user_counts[user_counts == 8].reset_index().groupby("behavior").agg({"user":set})["user"].values.tolist()))
