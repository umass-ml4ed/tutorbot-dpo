import os
import gc
import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seed_num: int):
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

def run_gc():
    torch.cuda.empty_cache()
    gc.collect()

def bool_type(x: str):
    return x != "0"

def get_checkpoint_path(model_name: str):
    os.makedirs("saved_models", exist_ok=True)
    return f"saved_models/{model_name}"

def get_model_file_suffix(args):
    if args.model_name:
        return args.model_name
    model_name = args.base_model.replace("/", "-")
    if args.detail_sys_prompt:
        model_name += "_dsp"
    return model_name
