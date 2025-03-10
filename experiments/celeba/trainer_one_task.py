import os
import re
import wandb
from pathlib import Path

from tqdm import trange
from argparse import ArgumentParser
import hydra
import logging

import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        
    def incr(self, y_, y_task):
        y_task = y_task.cpu().numpy()
        y_pred = (y_.detach().cpu().gt(0.5)).numpy()
        self.tp += (y_pred * y_task).sum()
        self.fp += (y_pred * (1 - y_task)).sum()
        self.fn += ((1 - y_pred) * y_task).sum()
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item()

@hydra.main(version_base='1.3', config_path="conf", config_name="config_singletask")
def main(cfg):

    # set random seed fopr reprobucibility
    torch.manual_seed(cfg.random_seed)

    # set device
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu")

    
    # Convert cfg.optimization.method to a Path object
    method_path = Path(cfg.optimization.method)  

    # set Checkpoint folders
    chk_path = Path(cfg.checkpoint_path)
    chk_path = chk_path / method_path
    chk_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoint are saved in: {chk_path.as_posix()}")
    
    # set Results folders
    res_path = Path(cfg.out_path) 
    res_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results are saved in: {res_path.as_posix()}")
    # we only train for specific task
    model = Network().to(device)
    
    train_set = CelebaDataset(data_dir=cfg.data_path, split='train')
    val_set = CelebaDataset(data_dir=cfg.data_path, split='val')
    test_set = CelebaDataset(data_dir=cfg.data_path, split='test')

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    epochs    = cfg.trainer.n_epochs
    metrics   = np.zeros([epochs], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    best_val_f1 = 0.0
    best_epoch = None
    for epoch in range(epochs):
        # training
        t0 = time.time()
        for x, y in tqdm.tqdm(train_loader):
            x = x.to(device)
            y_task = y[cfg.trainer.task].to(device)
            y_ = model(x, task=cfg.trainer.task)
            loss = loss_fn(y_, y_task)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t1 = time.time()

        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_task = y[cfg.trainer.task].to(device)
                y_ = model(x, task=cfg.trainer.task)
                loss = loss_fn(y_, y_task)
                metric.incr(y_, y_task)
        val_f1 = metric.result()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y_task = y[cfg.trainer.task].to(device)
                y_ = model(x, task=cfg.trainer.task)
                loss = loss_fn(y_, y_task)
                metric.incr(y_, y_task)
        test_f1 = metric.result()
        metrics[epoch] = test_f1
        t2 = time.time()
        logging.info(f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min")

        name = f"STL_sd{cfg.random_seed}_task{cfg.trainer.task}"
        save_path = res_path / f"{name}.stats"
        torch.save({"metric": metrics, "best_epoch": best_epoch, "best_val_f1":best_val_f1}, save_path)

        
       

if __name__ == "__main__":
    main()
