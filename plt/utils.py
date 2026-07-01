import os
import torch 
import torch.nn as nn
import random
import yaml
import math 
import numpy as np
from tqdm import tqdm 
from omegaconf import OmegaConf


def _pair(x):
    # Ensure values like kernel_size=3 become (3, 3)
    return x if isinstance(x, tuple) else (x, x)


def read_config(path_to_config):
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)  

    config = OmegaConf.create(config)
    
    return config


def list_partition(list_in, n):
    ls = list(list_in)
    random.shuffle(ls)
    return [ls[i::n] for i in range(n)]


def calc_mean_std(loader, get_hist=True):
    # var[X] = E[X**2] - E[X]**2
    stats = {}
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for images, _, _  in tqdm(loader):
        b, c, h, w = images.shape
        
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    stats["mean"], stats["std"] = mean, std

    return stats


def calc_mean_std_per_domain(loaders_dict):
    stats_per_domain = {}

    for domain, loader in loaders_dict.items():
        stats = calc_mean_std(loader)
        stats_per_domain[domain] = stats
    
    return stats_per_domain


def _make_unique_outdir(parent_dir: str, run_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    base_path = os.path.join(parent_dir, run_name)
    for suffix in range(10_000):
        path = base_path if suffix == 0 else f"{base_path}_{suffix}"
        try:
            os.makedirs(path, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique output directory under {parent_dir!r}.")



def random_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_by_groups(groups):
    unique_groups, unique_counts = torch.unique(groups, sorted=False, return_counts=True)
    group_indices = {}

    for group in unique_groups:
        group_indices[int(group)] = torch.nonzero(groups == group, as_tuple=True)[0]

    return group_indices, unique_groups, unique_counts


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    vpqt = torch.stack((v, p, q, t), dim=-3)

    # vpqt -> rgb mapping based on i
    select = torch.tensor([[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long)
    select = select.to(device=img.device, non_blocking=True)

    select = select[:, i]

    if select.ndim > 3:
        # if input.shape is (B, ..., C, H, W) then
        # select.shape is (C, B, ...,  H, W)
        # thus we move C axis to get (B, ..., C, H, W)
        select = select.moveaxis(0, -3)

    return vpqt.gather(-3, select)




