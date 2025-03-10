import os
import re
import wandb
from pathlib import Path

from omegaconf import DictConfig
from tqdm import trange
import logging
import hydra

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CelebaDataset
from models import Network
from experiments.utils import (
    extract_weight_method_parameters_from_cfg,
    get_device,
    set_logger,
    set_seed,
)
from methods.weight_methods import WeightMethods


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0.0 
        self.fp = 0.0 
        self.fn = 0.0 
        
    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds  = torch.stack(y_preds).detach() # (40, batch, 1)
        ys       = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds  = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1,2]) # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1,2])
        self.fn += ((1 - y_preds) * ys).sum([1,2])
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()

set_logger()

@hydra.main(version_base='1.3',config_path="conf", config_name="config")
def main(cfg: DictConfig):

    logging.info(""" 


 ██████╗███████╗██╗     ███████╗██████╗      █████╗ 
██╔════╝██╔════╝██║     ██╔════╝██╔══██╗    ██╔══██╗
██║     █████╗  ██║     █████╗  ██████╔╝    ███████║
██║     ██╔══╝  ██║     ██╔══╝  ██╔══██╗    ██╔══██║
╚██████╗███████╗███████╗███████╗██████╔╝    ██║  ██║
 ╚═════╝╚══════╝╚══════╝╚══════╝╚═════╝     ╚═╝  ╚═╝
                                                    


 """)
    # set random seed fopr reprobucibility
    set_seed(cfg.random_seed)

    # set device
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu")

    
    # Convert cfg.optimization.method to a Path object
    method_path = Path(cfg.optimization.method) 
    seed_path = Path(str(cfg.random_seed)) 

    # set Checkpoint folders
    chk_path = Path(cfg.checkpoint_path)
    chk_path = chk_path / method_path / seed_path 
    chk_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoint are saved in: {chk_path.as_posix()}")
    
    # set Results folders
    res_path = Path(cfg.out_path) 
    res_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results are saved in: {res_path.as_posix()}")

    # we only train for specific task
    model = Network().to(device)
    
    train_set = CelebaDataset(data_dir=cfg.data_path, split='train')
    val_set   = CelebaDataset(data_dir=cfg.data_path, split='val')
    test_set  = CelebaDataset(data_dir=cfg.data_path, split='test')

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    epochs    = cfg.trainer.n_epochs
    latest_epoch = 0

    metrics   = np.zeros([epochs, 40], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    # weight method
    logging.info(f"weight method: {cfg.optimization.method}")
    method_params = dict(cfg.optimization)
    method_params.pop("method", None)
    weight_method = WeightMethods(
        method= cfg.optimization.method, n_tasks=40, device=device, **method_params
    )

    best_val_f1 = 0.0
    best_epoch = None


    # load checkpoint if exist
    checkpoint_files = list(chk_path.glob("epoch-*.pth"))
    if len(checkpoint_files) != 0:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.search(r'epoch-(\d+)', x.stem).group(1)))
        # latest_epoch = int(latest_checkpoint.stem.split('-')[1])
    
        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint, weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        latest_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        # load metrics
        name = f"{cfg.optimization.method}_sd{cfg.random_seed}"
        save_path = res_path / f"{name}.stats"
        last_metric = torch.load(save_path)
        metrics = last_metric['metric']
        best_epoch = last_metric['best_epoch']
        best_val_f1 = last_metric['best_val_f1']
        
        logging.info(f"Resuming from checkpoint: {latest_checkpoint}, epoch {latest_epoch}")
 
    for epoch in trange(latest_epoch,epochs):
        # training
        model.train()
        t0 = time.time()
        
        for x, y in train_loader:
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
     
            losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
            optimizer.zero_grad()
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()
        t1 = time.time()


        # save checkpoint
        save_path = chk_path / f"chk_momentum{cfg.optimization.momentum}_similarity{cfg.optimization.gamma}_beta2{cfg.optimization.beta2}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, save_path)

        model.eval()
        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        val_f1 = metric.result()
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        test_f1 = metric.result()
        metrics[epoch] = test_f1

        t2 = time.time()
        logging.info(f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min")
        
        name = f"{cfg.optimization.method}_sd{cfg.random_seed}_momentum{cfg.optimization.momentum}_similarity{cfg.optimization.gamma}_beta2{cfg.optimization.beta2}"
        save_path = res_path / f"{name}.stats"
        torch.save({"metric": metrics, "best_epoch": best_epoch, "best_val_f1":best_val_f1}, save_path)


if __name__ == "__main__":
    main()
