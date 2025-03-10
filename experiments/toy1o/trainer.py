#!/usr/bin/env python

import time
import json
import numpy as np
import torch
import os
from tqdm import tqdm
import logging
from pathlib import Path

from problem import Toy


from methods.weight_methods import WeightMethods
from methods.utils import set_logger,conflict_grads,gradient_magnitude_similarity,curvarute_bounding_measure,convert_to_serializable


def trainer(cfg, device):
    n_tasks = cfg.interest_function.n_task
    F = Toy(scale=cfg.problem.scale,scale_both_losses=cfg.problem.scale_both_losses,Interest_function=cfg.interest_function.function)
    

    all_traj = dict()
    all_time = dict()

    if not cfg.plotSpecification.inits :
        inits = [torch.tensor([x, y], dtype=torch.float32) for x in torch.arange(-11, 11, 0.1) for y in torch.arange(-10, 10, 0.1)]
    else:
        inits = [torch.tensor(init, dtype=torch.float32) for init in cfg.plotSpecification.inits]
    

    for i, init in enumerate(inits):
        
        logging.info(f"point {i}/{len(inits)}")
        if not cfg.plotSpecification.inits :
            all_traj = dict()
            all_time = dict()
        traj = []
        grad = []
        weight = []
        x = init.clone()
        x = x.to(device)
        x.requires_grad = True
        
        method_params = dict(cfg.optimization)
        method_params.pop("method", None)

        method = WeightMethods(
            method= cfg.optimization.method,
            device=device,
            n_tasks=n_tasks,
            ** method_params
        )

        optimizer = torch.optim.Adam(params=[x], lr=1e-3)

        t0 = time.time()
        for it in tqdm(range(cfg.trainer.n_epochs)):
            traj.append(x.cpu().detach().numpy().copy())

            optimizer.zero_grad()
            f,g = F(x, True)
            grad.append(g)
            f = F(x, False)
            
            if "famo" in cfg.optimization.method and it > 0:
                with torch.no_grad():
                    method.method.update(f)

            _, extra_outputs = method.backward(
                losses=f.to(device),
                shared_parameters=(x,),
                task_specific_parameters=None,
                last_shared_parameters=None,
                representation=None,
            )
            weight.append(extra_outputs)
            optimizer.step()
        t1 = time.time()
        all_time[i] = t1-t0
        all_traj[i] = dict(init=init.cpu().detach().numpy().copy(), 
                        traj=np.array(traj),
                        grad=np.array(grad),
                        weight = weight)
        
        

    return all_traj, all_time
