import os
import re
import logging
import hydra
import wandb

from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from data import NYUv2
from models import SegNet, SegNetMtan
from utils import ConfMatrix, delta_fn, depth_error, normal_error,save_checkpoint,load_checkpoint
from experiments.utils import (
    set_logger,
    set_seed
)
from methods.weight_methods import WeightMethods

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


 


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    logging.info("""


███╗   ██╗██╗   ██╗██╗   ██╗    ██╗   ██╗██████╗ 
████╗  ██║╚██╗ ██╔╝██║   ██║    ██║   ██║╚════██╗
██╔██╗ ██║ ╚████╔╝ ██║   ██║    ██║   ██║ █████╔╝
██║╚██╗██║  ╚██╔╝  ██║   ██║    ╚██╗ ██╔╝██╔═══╝ 
██║ ╚████║   ██║   ╚██████╔╝     ╚████╔╝ ███████╗
╚═╝  ╚═══╝   ╚═╝    ╚═════╝       ╚═══╝  ╚══════╝
                                                 

                          
""" )
    
    # set wandb
    if cfg.logger.wandb:
        wandb.init(project=cfg.logger.project, config=OmegaConf.to_container(cfg, resolve=True))

    # set random seed fopr reprobucibility
    set_seed(cfg.random_seed)

    # device
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu")
    
    # Convert cfg.optimization.method to a Path object
    method_path = Path(cfg.optimization.method) 
    seed_path = Path(str(cfg.random_seed)) 
    
    # set Checkpoint folders
    chk_path = Path(cfg.checkpoint.path)
    chk_path = chk_path / method_path / seed_path
    chk_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoint are saved in: {chk_path.as_posix()}")
    
    # set Results folders
    res_path = Path(cfg.out_path) 
    res_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results are saved in: {res_path.as_posix()}")
    
    # Nets
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[cfg.model.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if cfg.trainer.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=cfg.data_path, train=True, augmentation=cfg.trainer.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=cfg.data_path, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=cfg.trainer.batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=cfg.trainer.batch_size, shuffle=False
    )

    # weight method
    method_params = dict(cfg.optimization)
    method_params.pop("method", None)
    weight_method = WeightMethods(
        cfg.optimization.method, n_tasks=3, device=device, **method_params
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=cfg.trainer.lr),
            # dict(params=weight_method.parameters(), lr=cfg.optimization.method_params_lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = cfg.trainer.n_epochs
    latest_epoch = 0
    custom_step = -1

    # training stat
    train_batch = len(train_loader)
    test_batch = len(test_loader)

    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    loss_list = []

    # evaluation stat
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # load checkpoint if exist
    checkpoint_files = list(chk_path.glob("chk_nyuv2*.pth"))
    if len(checkpoint_files) != 0:
        latest_epoch, custom_step, deltas, keys, avg_cost, loss_list = load_checkpoint(chk_path, model, optimizer, scheduler,device)

    # some extra statistics we save during training
    epoch_iter = trange(latest_epoch,epochs)
    

    for epoch in epoch_iter:
        cost = np.zeros(24, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()
            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                    calc_loss(train_pred[2], train_normal, "normal"),
                )
            )

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            # for record intermediate statistics
            loss_list.append(losses.detach().cpu())
            optimizer.step()


            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}, "
                f"Method: {extra_outputs['weights']}"
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()
        
        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )
            deltas[epoch] = test_delta_m

            # print results
            logging.info(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
            )
            logging.info(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} || "
                f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | {avg_cost[epoch, 18]:.4f} "
                f"{avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f} "
                f"| {test_delta_m:.3f}"
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
                wandb.log({"Train Normal Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Train Loss Mean": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Train Loss Med": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Train Loss <11.25": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Train Loss <22.5": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Train Loss <30": avg_cost[epoch, 11]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 12]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 13]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 14]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 15]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 16]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 17]}, step=epoch)
                wandb.log({"Test Normal Loss": avg_cost[epoch, 18]}, step=epoch)
                wandb.log({"Test Loss Mean": avg_cost[epoch, 19]}, step=epoch)
                wandb.log({"Test Loss Med": avg_cost[epoch, 20]}, step=epoch)
                wandb.log({"Test Loss <11.25": avg_cost[epoch, 21]}, step=epoch)
                wandb.log({"Test Loss <22.5": avg_cost[epoch, 22]}, step=epoch)
                wandb.log({"Test Loss <30": avg_cost[epoch, 23]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)



            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",
                "Train Normal Loss",
                "Train Loss Mean",
                "Train Loss Med",
                "Train Loss <11.25",
                "Train Loss <22.5",
                "Train Loss <30",

                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
                "Test Normal Loss",
                "Test Loss Mean",
                "Test Loss Med",
                "Test Loss <11.25",
                "Test Loss <22.5",
                "Test Loss <30"
            ]

            # save checkpoints
            if cfg.checkpoint.save:
                save_checkpoint(chk_path, model, optimizer, scheduler, epoch, custom_step, deltas, keys, avg_cost, loss_list)
            
            # save results
            name = f"{cfg.optimization.method}_sd{cfg.random_seed}_momentum{cfg.optimization.momentum}_similarity{cfg.optimization.gamma}_beta2{cfg.optimization.beta2}"
            save_path = res_path / f"{name}.stats"
            torch.save({
                "delta_m": deltas,
                "keys": keys,
                "avg_cost": avg_cost,
                "losses": loss_list,
            }, save_path)

            
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()

