#!/usr/bin/env python

import time
import numpy as np
import torch
from tqdm import tqdm

from problem import Toy


from methods.weight_methods import WeightMethods

def trainer(cfg, device):
    n_tasks = cfg.interest_function.n_task
    F = Toy(scale=cfg.problem.scale,Interest_function=cfg.interest_function.function)


    all_traj = dict()
    all_time = dict()

    # the initial positions
    # Convert the list of lists into PyTorch tensors
    if not cfg.plotSpecification.inits :
        inits = [torch.tensor([x, y], dtype=torch.float32) for x in torch.arange(-9, 9, 0.05) for y in torch.arange(-12, 11, 0.05)]
    else:
        inits = [torch.tensor(init, dtype=torch.float32) for init in cfg.plotSpecification.inits]
    

    for i, init in enumerate(inits):
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

            if "famo" in cfg.optimization.method and it > 0:
                with torch.no_grad():
                    method.method.update(f)

            grad.append(g)
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
