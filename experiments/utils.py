
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods.weight_methods import METHODS


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """
    Reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
                max_norm=args.max_norm,
            ),
            stl=dict(main_task=args.main_task),
            dwa=dict(temp=args.dwa_temp),
            cagrad=dict(c=args.c, max_norm=args.max_norm),
            log_cagrad=dict(c=args.c, max_norm=args.max_norm),
            famo=dict(gamma=args.gamma,
                      w_lr=args.method_params_lr,
                      max_norm=args.max_norm),
        )
    )
    return weight_methods_parameters

def extract_weight_method_parameters_from_cfg(method_cfg):

    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=method_cfg.update_weights_every,
                optim_niter=method_cfg.nashmtl_optim_niter,
                max_norm=method_cfg.max_norm,
            ),
            stl=dict(main_task=method_cfg.main_task),
            dwa=dict(temp=method_cfg.dwa_temp),
            cagrad=dict(c=method_cfg.c, max_norm=method_cfg.max_norm),
            log_cagrad=dict(c=method_cfg.c, max_norm=method_cfg.max_norm),
            famo=dict(
                gamma=method_cfg.gamma,
                w_lr=method_cfg.method_params_lr,
                max_norm=method_cfg.max_norm,
            ),
        )
    )
    return weight_methods_parameters


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
