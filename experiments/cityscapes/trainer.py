import os
import re
import logging
import hydra
import wandb
import json

from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from data import Cityscapes
from models import SegNet, SegNetMtan
from utils import ConfMatrix, delta_fn, depth_error
from experiments.utils import (
    set_logger,
    set_seed,
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

    return loss



@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig):

    logging.info(""" 


 ██████╗██╗████████╗██╗   ██╗███████╗ ██████╗ █████╗ ██████╗ ███████╗
██╔════╝██║╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝
██║     ██║   ██║    ╚████╔╝ ███████╗██║     ███████║██████╔╝█████╗  
██║     ██║   ██║     ╚██╔╝  ╚════██║██║     ██╔══██║██╔═══╝ ██╔══╝  
╚██████╗██║   ██║      ██║   ███████║╚██████╗██║  ██║██║     ███████╗
 ╚═════╝╚═╝   ╚═╝      ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝
                                                                     


 """)
    # set wandb
    if cfg.logger.wandb:
        wandb.init(project=cfg.logger.project, config=OmegaConf.to_container(cfg, resolve=True),name = f"{cfg.optimization.method}_sd{cfg.random_seed}")

    # set device
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu")

    # set random seed fopr reprobucibility
    set_seed(cfg.random_seed)

    # Convert cfg.optimization.method to a Path object
    method_path = Path(cfg.optimization.method) 
    seed_path = Path(str(cfg.random_seed)) 
    
    # set Checkpoint folders
    chk_path = Path(cfg.checkpoint.path)
    chk_path = chk_path / method_path / seed_path / Path(f'momentum{cfg.optimization.momentum}_similarity{cfg.optimization.gamma}_beta2{cfg.optimization.beta2}')
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
        "Applying data augmentation on cityscapes."
        if cfg.trainer.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(
        root=cfg.data_path, train=True, augmentation=cfg.trainer.apply_augmentation
    )
    cityscapes_test_set = Cityscapes(root=cfg.data_path, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_train_set, batch_size=cfg.trainer.batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_test_set, batch_size=cfg.trainer.batch_size, shuffle=False
    )

    # weight method
    method_params = dict(cfg.optimization)
    method_params.pop("method", None)

    weight_method = WeightMethods(
        method=cfg.optimization.method, n_tasks=2, device=device, **method_params
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=cfg.trainer.lr),
            
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    epochs = cfg.trainer.n_epochs
    latest_epoch = 0
    custom_step = -1

    train_batch = len(train_loader)
    test_batch = len(test_loader)

    # training stat
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    alpha = torch.tensor([1.,1.],device=device)
    loss_list = []

    # evaluation stat
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs,], dtype=np.float32)
    

    # load checkpoint if exist
    checkpoint_files = list(chk_path.glob("chk_cityscape.pth"))
    if len(checkpoint_files) != 0:
        checkpoint_files = sorted(chk_path.glob("chk_cityscape.pth"), reverse=True)
        logging.info(checkpoint_files[0])
        latest_checkpoint = chk_path.glob("chk_cityscape.pth") #checkpoint_files[0] #max(checkpoint_files, key=lambda x: int(re.search(r'epoch-(\d+)', x.stem).group(1)))
        # latest_epoch = int(latest_checkpoint.stem.split('-')[1])
    
        # Load the checkpoint
        try:
            checkpoint = torch.load(chk_path.glob("chk_cityscape.pth"),map_location='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        latest_epoch = checkpoint['epoch']
        custom_step = checkpoint['custom_step']
        deltas = checkpoint['metrics']['deltas']
        keys = checkpoint['metrics']['keys']
        avg_cost = checkpoint['metrics']['avg_cost']
        loss_list = checkpoint['metrics']['loss_list']
        
        logging.info(f"Resuming from checkpoint: {latest_checkpoint}, epoch {latest_epoch}")

    # some extra statistics we save during training
    
    epoch_iter = trange(latest_epoch,epochs)

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)
        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth = train_depth.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )
         
            loss, extra_outputs = weight_method.backward(
                losses=losses*alpha,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )
            loss_list.append(losses.detach().cpu())
            optimizer.step()

            if any(method in cfg.optimization.method for method in ["famo"]):
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                        (
                            calc_loss(train_pred[0], train_label, "semantic"),
                            calc_loss(train_pred[1], train_depth, "depth"),
                        )
                    )
                    weight_method.method.update(new_losses.detach())

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f} "
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
                test_data, test_label, test_depth = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [7, 8, 10, 11]]
            )
            deltas[epoch] = test_delta_m

            # print results
            logging.info(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
            )
            logging.info(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"TEST: {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} | "
                f"{avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}"
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
                
                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)



            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",

                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
            ]

            # save checkpoints
            if cfg.checkpoint.save:
                save_path = chk_path / f"chk_cityscape.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'custom_step': custom_step,
                    'metrics': {
                        "delta_m": deltas,
                        "keys": keys,
                        "avg_cost": avg_cost,
                        "losses": loss_list,}
                    }, save_path)
            
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

   