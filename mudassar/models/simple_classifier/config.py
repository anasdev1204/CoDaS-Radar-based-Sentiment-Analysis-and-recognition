import random
from copy import deepcopy
import json
import re


def baseline_model_config():
    radar_config = {
        "input_dim": 4,
        "output_dim": 1,
        "point_cloud_encoder_kwargs": {"embedding_size": 32, "num_layers": 2, "num_heads": 4},
        "temporal_encoder_kwargs": {"embedding_size": 128, "num_layers": 2, "num_heads": 8},
    }
    infrared_config = {
        "input_dim": 21,
        "output_dim": 1,
        "downsampler_kwargs": {"output_dim": 32, "hidden_dims": (32, 32)},
        "temporal_encoder_kwargs": {"embedding_size": 128, "num_layers": 2, "num_heads": 8},
    }
    fancy_infrared_config = {
        "input_dim": 7,
        "output_dim": 1,
        "downsampler_kwargs": {"output_dim": 32, "hidden_dims": (16, 32)},
        "point_cloud_encoder_kwargs": {"embedding_size": 32, "num_layers": 2, "num_heads": 4},
        "temporal_encoder_kwargs": {"embedding_size": 128, "num_layers": 2, "num_heads": 8},
    }
    return {
            "radar": radar_config,
            "infrared": infrared_config,
            "fancy_infrared": fancy_infrared_config,
    }

def model_param_variants():
    return {
        "radar         .point_cloud_encoder_kwargs.embedding_size": [16, 32, 64, 128, 256],
        "fancy_infrared.point_cloud_encoder_kwargs.embedding_size": [16, 32, 64, 128, 256],
        "radar         .point_cloud_encoder_kwargs.num_layers": [1, 2, 3, 4],
        "fancy_infrared.point_cloud_encoder_kwargs.num_layers": [1, 2, 3,  ],
        "radar         .point_cloud_encoder_kwargs.num_heads": [4, 8, 16, 32],
        "fancy_infrared.point_cloud_encoder_kwargs.num_heads": [4, 8, 16, 32],
        "radar         .point_cloud_encoder_kwargs.pooling_strategy": ["mean"],
        "fancy_infrared.point_cloud_encoder_kwargs.pooling_strategy": ["mean"],
        "fancy_infrared.point_cloud_encoder_kwargs.use_pos_emb": [True, False],

        "radar         .temporal_encoder_kwargs.embedding_size": [64, 128, 256, 512, 1024],
        "infrared      .temporal_encoder_kwargs.embedding_size": [64, 128, 256, 512, 1024],
        "fancy_infrared.temporal_encoder_kwargs.embedding_size": [64, 128, 256, 512, 1024],
        "radar         .temporal_encoder_kwargs.num_layers": [1, 2, 3, 4, 5, 6],
        "infrared      .temporal_encoder_kwargs.num_layers": [1, 2, 3, 4, 5, 6],
        "fancy_infrared.temporal_encoder_kwargs.num_layers": [1, 2, 3, 4, 5, 6],
        "radar         .temporal_encoder_kwargs.num_heads": [4, 8, 16, 32, 64],
        "infrared      .temporal_encoder_kwargs.num_heads": [4, 8, 16, 32, 64],
        "fancy_infrared.temporal_encoder_kwargs.num_heads": [4, 8, 16, 32, 64],
        "radar         .temporal_encoder_kwargs.pooling_strategy": ["mean", "eos"],
        "infrared      .temporal_encoder_kwargs.pooling_strategy": ["mean", "eos"],
        "fancy_infrared.temporal_encoder_kwargs.pooling_strategy": ["mean", "eos"],

        "infrared      .downsampler_kwargs.output_dim": [None, 16, 32, 64, 128, 256],
        "fancy_infrared.downsampler_kwargs.output_dim": [      16, 32, 64, 128, 256],
        "infrared      .downsampler_kwargs.hidden_dims": [(), (16,), (32,), (64,), (128,), (16, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (16, 32, 64), (32, 64, 64), (32, 64, 128)],
        "fancy_infrared.downsampler_kwargs.hidden_dims": [(), (16,), (32,), (64,), (128,), (16, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (16, 32, 64), (32, 64, 64), (32, 64, 128)],
    }

def filter_params(params: dict, model_type: str = "", n: int = None):
    filtered_params = {}
    for key, value in params.items():
        if key.startswith(model_type):
            filtered_params[key.replace(" ", "")] = value
    if n is not None:
        filtered_params = list(filtered_params.items())
        random.shuffle(filtered_params)
        filtered_params = dict(sorted(filtered_params[:n]))
    return filtered_params

def random_param_values(params: dict=None, model_type: str = "", n: int = None):
    if params is None:
        params = model_param_variants()
    variant = {}
    for key, values in params.items():
        variant[key] = random.choice(values)
    if model_type or n is not None:
        variant = filter_params(variant, model_type=model_type, n=n)
    return variant

def apply_params_to_config(config: dict, params: dict):
    config = deepcopy(config)
    for key, value in params.items():
        keys = [k.strip() for k in key.split(".")]
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config

def random_model_variant(base_config: dict = None, model_type: str = "", n: int = None):
    config = baseline_model_config() if base_config is None else deepcopy(base_config)
    params = random_param_values(model_param_variants(), model_type=model_type, n=n)
    new_config = apply_params_to_config(config, params)
    if model_type:
        new_config = {model_type: new_config[model_type]}
        config = {model_type: config[model_type]}
    return new_config, config, params

def mutation_id(params: dict):
    def minify_key(key):
        return ".".join("".join(w[0] for w in k.strip().split("_") if w not in ("kwargs",)) for k in key.split("."))

    def minify_value(value):
        if isinstance(value, (bool, type(None), str)):
            return str(value)[0].upper()
        return value

    if not params:
        return "baseline"

    if isinstance(params, str):
        params = json.loads(params)
    config = apply_params_to_config({}, {minify_key(key): minify_value(v) for key, v in sorted(params.items())})
    string = json.dumps(config, sort_keys=True)
    string = re.sub(r'[: "]', "", string)
    string = re.sub(r"(?<=[\dECMTFN\}\]]),(?=[a-z])", "", string)
    return string[1:-1]
