import random
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd
import torch
import numpy as np

from models.simple_classifier import config as C, train as T, model as M
from data_readers import data_loader as D
import genetic_search as G

@dataclass
class ExperimentLog:
    model_type: str = ""
    n_params: int = 0
    test_f1: float = 0.0
    labels: list[str] = field(default_factory=list)
    train_loss: float = 0.0
    valid_loss: float = 0.0
    test_users: list[str] = field(default_factory=list)
    valid_users: list[str] = field(default_factory=list)
    mutations: str = ""
    lr: float = 0.0
    epochs: int = 0
    batch_size: int = 0
    augment_rate: float = 0.0
    infer_times: list[float] = field(default_factory=list)
    train_times: list[float] = field(default_factory=list)

    @property
    def mutation_id(self) -> str:
        if not self.mutations:
            return ""
        return C.mutation_id(self.mutations)

    @property
    def output_id(self) -> str:
        if not self.labels:
            return ""
        return "".join(self.labels)

    @property
    def model_id(self) -> str:
        return f"{self.mutation_id}_{self.n_params}_{self.output_id}"

    def to_dict(self):
        return {k: v for k, v in list(self.__dict__.items())+[("model_id", self.model_id)] if v}

    def save(self, filename: str):
        with open(filename, "a", newline="") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

    def copy(self, **kwargs):
        new = deepcopy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new

def split_data(df:pd.DataFrame, valid_users, test_users):
    valid_mask = df.user.isin(valid_users)
    test_mask = df.user.isin(test_users)
    valid_df = df[valid_mask]
    test_df = df[test_mask]
    train_df = df[~valid_mask & ~test_mask]
    # print(f"{train_df.shape=} | {valid_df.shape=} | {test_df.shape=}")
    return train_df, valid_df, test_df

def get_data_for_experiment(df:pd.DataFrame, labels=None, n_valid_users=2, n_test_users=4, valid_users=None, test_users=None, common_users=None, log=None):
    if not log:
        log = ExperimentLog()
    if not log.labels:
        if labels is None:
            raise ValueError("Either `log.labels` or `labels` must be provided.")
        log.labels = sorted(labels)
    exp_data = df[df.behavior.isin(log.labels)].copy()
    exp_data["label"] = exp_data.behavior.apply(log.labels.index)
    if valid_users and not log.valid_users:
        log.valid_users = sorted(valid_users)
    if test_users and not log.test_users:
        log.test_users = sorted(test_users)
    if not log.valid_users or not log.test_users:
        if common_users is None:
            common_users = D.get_common_users(df)
        random.shuffle(common_users)
        if not log.valid_users:
            log.valid_users = sorted(common_users[:n_valid_users])
        if not log.test_users:
            log.test_users  = sorted(common_users[-n_test_users:])
        if len(set(log.valid_users) & set(log.test_users)) > 0:
            raise ValueError(f"Valid and test users must be disjoint. Found overlap: {set(log.valid_users) & set(log.test_users)}")
    train_df, valid_df, test_df = split_data(exp_data, log.valid_users, log.test_users)
    return train_df, valid_df, test_df, log

def run_one_experiment(model: torch.nn.Module, train_dataset: D.BehaviorDataset, valid_dataset: D.BehaviorDataset, test_dataset: D.BehaviorDataset, log=None, verbose=False, device="cpu", max_epochs=100, patience=5):
    if log is None:
        log = ExperimentLog(labels=train_dataset.label_names, valid_users=valid_dataset.users or [], test_users=test_dataset.users or [])

    if not log.lr:
        log.lr = 1e-4
    if not log.batch_size:
        log.batch_size = 16

    history = T.fit(model, train_dataset, valid_dataset, lr=log.lr, batch_size=log.batch_size, verbose=verbose, collate_fn=D.BehaviorDataset.collate_fn, pin_memory=True, device=device, epochs=max_epochs, patience=patience)
    log.train_loss = history["train_loss"][history["checkpoint_epoch"]]
    log.valid_loss = history["valid_loss"][history["checkpoint_epoch"]]
    log.epochs = history["checkpoint_epoch"] + 1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=log.batch_size, shuffle=False, collate_fn=D.BehaviorDataset.collate_fn, pin_memory=True, num_workers=0)
    log.test_f1 = float(T.evaluate_model(model.eval(), test_loader, device=device)[2])

    return log

def load_model_df(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['model_type', 'n_params', 'mutations', 'infer_times', 'train_times', 'bucket', 'modality'])

    model_df = pd.read_csv(path)
    model_df["bucket"] = model_df.n_params.apply(lambda x: np.log(x)/np.log(47)).round(1)
    model_df["modality"] = model_df.model_type.str.split("_").str[-1]
    return model_df

def load_exp_df(path):
    if os.path.exists(path):
        return pd.read_json(path, lines=True)
    return pd.DataFrame(columns=list(ExperimentLog().__dict__.keys()) + list(ExperimentLog().to_dict().keys()))

def _update_input_output_dim(mutations: str, labels: list) -> str:
    mut = json.loads(mutations)
    model_type = next(iter(mut)).split(".")[0]
    n_out = len(labels)
    n_in = 6*7 if any(l.startswith(("A", "E")) for l in labels) else 3*7

    if model_type == "infrared":
        mut[model_type+".input_dim"] = n_in
    mut[model_type+".output_dim"] = (1 if n_out==2 else n_out)
    return json.dumps(mut, sort_keys=True)

def select_mutations_grid(model_grouped_df, logs_file_path:str, labels):

    log = ExperimentLog(labels=sorted(labels))
    exp_df = load_exp_df(logs_file_path)
    processed = set(exp_df.mutations.tolist()) if not exp_df.empty else set()

    for _ in range(10):
        df: pd.DataFrame = model_grouped_df.sample(1).sort_values(["modality", "bucket"]).reset_index(drop=True).copy()
        df["mutations"] = df["mutations"].apply(lambda m: _update_input_output_dim(m, labels))
        df = df[~df["mutations"].isin(processed)]
        if not df.empty:
            return [log.copy(mutations=row["mutations"], model_type=row["model_type"]) for _, row in df.iterrows()]

    print(f"Warning: No new mutations found for labels {labels}. Returning empty list.")
    return []

def select_mutations_genetic(logs_file_path:str, labels):
    log = ExperimentLog(labels=sorted(labels))
    exp_df = load_exp_df(logs_file_path)
    processed = set(exp_df.mutations.tolist()) if not exp_df.empty else set()
    mutations = G.get_offsprings(exp_df, n_offsprings=5)
    return [log.copy(mutations=_update_input_output_dim(mut, labels), model_type=next(iter(json.loads(mut))).split(".")[0]) for modality, muts in mutations.items() for mut in muts if mut not in processed]

def run_experiments(labels, batch_size=16, lr=1e-4, dataset_dir="RF-Behavior", experiments_dir="experiments", logs_filename="experiment_logs.jsonl", model_variants_path="model_variants.csv", radar_bin_fps=18.7, device="cpu", max_epochs=100, patience=5, augment_rate=0.0, use_genetic=False):
    data_df = D.find_available_files(dataset_dir)
    print(f"{data_df.shape=} NA:", data_df.isna().sum(0).filter(regex=r".*_path").to_dict())
    # print(data_df.sample(1))
    mask = data_df.campaign == "C3"
    data_df = pd.concat([data_df[~mask].reset_index(drop=True).copy()] + [data_df[mask].reset_index(drop=True).copy() for _ in range(8)], ignore_index=True)
    print(f"{data_df.shape=} NA:", data_df.isna().sum(0).filter(regex=r".*_path").to_dict())

    common_users = D.get_common_users(data_df)
    print(f"{len(common_users)=}:{common_users}")

    logs_path = os.path.join(experiments_dir, logs_filename)
    if use_genetic:
        logs = select_mutations_genetic(logs_path, labels)
    else:
        model_df = load_model_df(model_variants_path)
        # model_df = model_df[~((model_df.model_type == "radar") & model_df.mutations.str.contains(r'"eos"'))].copy()
        # model_df = model_df[ (model_df.model_type == "infrared")].copy()
        # model_df = model_df[~(model_df.mutations.str.contains(r'null'))].copy()
        # model_df = model_df[~(model_df.mutations.str.contains(r': \[\]'))].copy()
        print(f"{model_df.shape=} | buckets per modality:", model_df.groupby("modality").bucket.nunique().to_dict())
        logs = select_mutations_grid(model_df.groupby(["modality", "bucket"]), logs_path, labels)

    mod_to_log = {"radar": [], "infrared": []}
    for l in logs:
        mod_to_log[l.model_type.split("_")[-1]].append(l)
    print("modality_to_logs:", {k:len(v) for k,v in mod_to_log.items()})

    for modality, logs in mod_to_log.items():
        # if modality=="radar":
        #     continue
        prev = None
        for i, log in enumerate(logs[:10]):
            if augment_rate:
                log.augment_rate = augment_rate
            if prev is None or log.labels != prev.labels:
                train_df, valid_df, test_df, log = get_data_for_experiment(data_df.dropna(subset=[modality+"_path"]), labels=labels, common_users=common_users, log=log)
                train_dataset = D.BehaviorDataset(train_df, modality=modality, radar_bin_fps=radar_bin_fps, augment_rate=log.augment_rate).preload()
                valid_dataset = D.BehaviorDataset(valid_df, modality=modality, radar_bin_fps=radar_bin_fps).preload()
                test_dataset  = D.BehaviorDataset(test_df,  modality=modality, radar_bin_fps=radar_bin_fps).preload()
                print(f"Loaded data for {modality=} | {len(train_dataset)=}, {len(valid_dataset)=}, {len(test_dataset)=} | {valid_dataset.users=}, {test_dataset.users=}")
            else:
                log.valid_users = deepcopy(prev.valid_users)
                log.test_users = deepcopy(prev.test_users)

            if not log.lr:
                log.lr = lr
                print("LR set in log:", log.lr)
            if not log.batch_size:
                log.batch_size = batch_size
                print("Batch size set in log:", log.batch_size)

            model = M.get_model_from_mutations(log.mutations, seed=0).to(device)
            log.n_params = model.n_params

            print(f"[{i+1}] Running experiment for {type(model).__name__} with {model.n_params} parameters and config={json.dumps(model.config, indent=2)}")
            log = run_one_experiment(model, train_dataset, valid_dataset, test_dataset, log, verbose=True, device=device, max_epochs=max_epochs, patience=patience)
            print("Experiment completed.")

            if hasattr(model, "save") and callable(model.save):
                model.save(os.path.join(experiments_dir, log.model_id+".pt"))
            log.save(os.path.join(experiments_dir, logs_filename))
            _simplified_log_dict = {k:v for k, v in log.to_dict().items() if k!='mutations'}
            print(f"Experiment log saved to {os.path.join(experiments_dir, logs_filename)}: {_simplified_log_dict}")
            print("="*50, "\n\n")

            prev = log.copy()
