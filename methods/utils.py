#!/usr/bin/env python

##################################################################
# Adaptation of code from: https://github.com/Cranial-XIX/CAGrad #
##################################################################

import torch
import numpy as np
from scipy.optimize import minimize_scalar

from matplotlib import pyplot as plt
from matplotlib import path as mpath

import seaborn as sns
from PIL import Image

import logging
from tqdm import tqdm


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

def conflict_grads(grads):
    g1 = grads[:, 0]  # First column (gradient for task 1)
    g2 = grads[:, 1]  # Second column (gradient for task 2)
    return torch.dot(g1, g2)/(torch.norm(g1)*torch.norm(g2))


def gradient_magnitude_similarity(grads):
    g1 = grads[:, 0]  # First column (gradient for task 1)
    g2 = grads[:, 1]  # Second column (gradient for task 2)

    norm_gi = torch.norm(g1)
    norm_gj = torch.norm(g2)
    return (2 * norm_gi * norm_gj) / (norm_gi**2 + norm_gj**2)

def curvarute_bounding_measure(grads):
    g1 = grads[:, 0]
    g2 = grads[: ,1]

    # Calculate the cosine of the angle between g1 and g2
    cos_phi_12 = conflict_grads(grads)

    # Calculate the components of ξ(g1, g2)
    cos2_phi_12 = cos_phi_12 ** 2
    norm_diff_squared = torch.norm(g1 - g2) ** 2
    norm_sum_squared = torch.norm(g1 + g2) ** 2

    # Calculate ξ(g1, g2)
    return (1 - cos2_phi_12) * (norm_diff_squared / norm_sum_squared)


def convert_to_serializable(obj):
    """
    Recursively convert NumPy arrays, PyTorch tensors, and other non-serializable objects
    in a nested dictionary or list structure to JSON-serializable formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    else:
        return obj
