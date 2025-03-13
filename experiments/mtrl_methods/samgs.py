from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import time
import torch
from omegaconf import OmegaConf

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.utils.types import ConfigType, TensorType
#from mtrl.agent.mgda import MinNormSolver
from scipy.optimize import minimize, Bounds, minimize_scalar

def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device

def apply_vector_grad_to_parameters(
    vec: TensorType, parameters: Iterable[TensorType], accumulate: bool = False
):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (
                param.grad + vec[pointer : pointer + num_param].view_as(param).data
            )
        else:
            param.grad = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

class Agent(grad_manipulation_agent.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        agent_cfg: ConfigType,
        multitask_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Regularized gradient algorithm."""
        agent_cfg_copy = deepcopy(agent_cfg)

        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"
        OmegaConf.set_struct(agent_cfg_copy, True)

        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg_copy,
            device=device,
        )
        self.agent._compute_gradient = self._compute_gradient
        self._rng = np.random.default_rng()

        self.b1 = agent_cfg_copy['samgs_momentum']
        self.b2 = agent_cfg_copy['samgs_beta2']
        self.gamma = agent_cfg_copy['samgs_gamma']

        self.m = torch.zeros(1,1,device=device)
        self.v = torch.zeros(1,device=device)
        self.t = 0


        self.wi_map = {}
        self.num_param_block = -1
        self.conflicts = []
        self.last_w = None
        self.save_target = 500000

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        task_loss = self._convert_loss_into_task_loss(
            loss=loss, env_metadata=env_metadata
        )
        num_tasks = task_loss.shape[0]
        grad = []

        for index in range(num_tasks):
            grad.append(
                tuple(
                    _grad.contiguous()
                    for _grad in torch.autograd.grad(
                        task_loss[index],
                        parameters,
                        retain_graph=(retain_graph or index != num_tasks - 1),
                        allow_unused=allow_unused,
                    )
                )
            )

        grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )  

        regularized_grad = self.samgs(grad_vec)
        apply_vector_grad_to_parameters(regularized_grad, parameters)

    def mean_gradient_similarity(self,grad_vec):
        """
        Computes mean gradient similarity efficiently.
        grad_vec: Tensor of shape [num_tasks, dim]
        """
        mg = torch.linalg.norm(grad_vec, dim=1)  

        mg_i = mg.unsqueeze(0) 
        mg_j = mg.unsqueeze(1)  

        numerator = 2 * mg_i * mg_j
        denominator = mg_i**2 + mg_j**2

        num_elements = grad_vec.shape[0]
        triu_indices = torch.triu_indices(num_elements, num_elements, offset=1)

        similarities = (numerator / denominator)[triu_indices[0], triu_indices[1]]

        mean_similarity = similarities.mean()

        return mean_similarity, similarities


    def samgs(self, grad_vec):
        """
        grad_vec: [num_tasks, dim]
        """
        self.t += 1
        w = grad_vec.detach()  
        original_dim = w.shape[1]

        mean_sim, sim_vec = self.mean_gradient_similarity(w)

        if w.shape[1] < self.m.shape[1]:
            self.m = self.m[:, :original_dim]  
        elif w.shape[1] > self.m.shape[1]:
            padding_m = torch.ones((self.m.shape[0], w.shape[1] - self.m.shape[1]), device=w.device)  
            self.m = torch.cat([self.m, padding_m], dim=1)

        self.m = self.b1*self.m + (1-self.b1)*w
        self.v = self.b2*self.v + ((1-self.b2)*(1-mean_sim)**2)+1e-8
        mhat = self.m/(1-self.b1**self.t)
        vhat = self.v/(1-self.b2**self.t)

        if ((sim_vec < self.gamma).any()):
            l2_norms = torch.norm(w, dim=1, p=2)
            scaled_w_l2 = grad_vec / l2_norms.unsqueeze(1)
            scaling_factor = l2_norms.mean()
            adjusted_g_l2 = scaled_w_l2 * scaling_factor
            g = adjusted_g_l2.sum(0)
        else:
            g=(grad_vec*abs(mhat)/(torch.sqrt(vhat))).sum(0)
        return g
