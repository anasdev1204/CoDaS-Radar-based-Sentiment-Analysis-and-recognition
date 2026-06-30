from time import time
import torch

from . import model as M

def elapsed_time(start, end):
    return round((end - start) * 1000, 3)

def measure_model_inference_times(model: torch.nn.Module, x: torch.Tensor, reps=10):
    model.eval()
    inference_times = []
    with torch.no_grad():
        for _ in range(reps):
            infer_start = time()
            y = model(x)
            infer_end = time()
            inference_times.append(elapsed_time(infer_start, infer_end))
    return inference_times

def measure_model_train_times(model: torch.nn.Module, x: torch.Tensor, reps=10, y: torch.Tensor=None, optimizer: torch.optim.Optimizer=None):
    if y is None:
        y = torch.zeros(x.shape[0], device=x.device)
    model.train()
    train_times = []
    for _ in range(reps):
        train_start = time()
        if optimizer is not None:
            optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        if optimizer is not None:
            optimizer.step()
        train_end = time()
        train_times.append(elapsed_time(train_start, train_end))
    return train_times

def get_dummy_xy(seed=0, device= "cpu"):
    torch.manual_seed(seed)
    x = {
        "radar":       torch.randn(16, 24, 64, 4).to(device), # 21+6+10 = 37
        # "infrared-c1": torch.randn(16, 256, 3, 7).to(device), # 21
        "infrared": torch.randn(16, 256, 6, 7).to(device), # 6+10 = 16
    }
    y = torch.randint(0, 2, (16,), device=device)
    return x, y

def run_model_timer(model: torch.nn.Module, model_type: str, config=None, dummy_inputs=None, dummy_labels=None, multiplier=1):
    if dummy_inputs is None or dummy_labels is None:
        dummy_inputs, dummy_labels = get_dummy_xy(seed=0, device=model.device)

    # print("infer")
    if model_type=="radar":
        infer_ts = measure_model_inference_times(model, dummy_inputs["radar"], reps=7*multiplier)
    elif model_type=="infrared":
        infer_ts  = measure_model_inference_times(model, dummy_inputs["infrared"][..., :3, :], reps=4*multiplier)
        if config:
            model2 = M.get_model(model_type, {**config[model_type], "input_dim": 6*7}).to(model.device)
            infer_ts += measure_model_inference_times(model2, dummy_inputs["infrared"], reps=3*multiplier)
    elif model_type=="fancy_infrared":
        infer_ts  = measure_model_inference_times(model, dummy_inputs["infrared"][..., :3, :], reps=4*multiplier)
        infer_ts += measure_model_inference_times(model, dummy_inputs["infrared"], reps=3*multiplier)

    # print("train")
    optim = torch.optim.AdamW(model.parameters())
    if model_type=="radar":
        train_ts = measure_model_train_times(model, dummy_inputs["radar"], reps=7*multiplier, y=dummy_labels, optimizer=optim)
    elif model_type=="infrared":
        train_ts  = measure_model_train_times(model, dummy_inputs["infrared"][..., :3, :], reps=4*multiplier, y=dummy_labels, optimizer=optim)
        if config:
            optim = torch.optim.AdamW(model2.parameters())
            train_ts += measure_model_train_times(model2, dummy_inputs["infrared"], reps=3*multiplier, y=dummy_labels, optimizer=optim)
    elif model_type=="fancy_infrared":
        train_ts  = measure_model_train_times(model, dummy_inputs["infrared"][..., :3, :], reps=4*multiplier, y=dummy_labels, optimizer=optim)
        train_ts += measure_model_train_times(model, dummy_inputs["infrared"], reps=3*multiplier, y=dummy_labels, optimizer=optim)

    return infer_ts, train_ts