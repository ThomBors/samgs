#!/usr/bin/env python

import os
import json
import hydra
import pickle


import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import plot_2d_pareto,plot_contour,plot3d,plot_oneFunction_contour,plot3D_OneFunction,create_gif_from_pareto
from methods.utils import set_logger,conflict_grads,gradient_magnitude_similarity,curvarute_bounding_measure,convert_to_serializable

from trainer import trainer


set_logger()

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg):
    logging.info(""" 


████████╗ ██████╗ ██╗   ██╗    ██████╗ ██████╗  ██████╗ ██████╗ ██╗     ███████╗███╗   ███╗
╚══██╔══╝██╔═══██╗╚██╗ ██╔╝    ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██║     ██╔════╝████╗ ████║
   ██║   ██║   ██║ ╚████╔╝     ██████╔╝██████╔╝██║   ██║██████╔╝██║     █████╗  ██╔████╔██║
   ██║   ██║   ██║  ╚██╔╝      ██╔═══╝ ██╔══██╗██║   ██║██╔══██╗██║     ██╔══╝  ██║╚██╔╝██║
   ██║   ╚██████╔╝   ██║       ██║     ██║  ██║╚██████╔╝██████╔╝███████╗███████╗██║ ╚═╝ ██║
   ╚═╝    ╚═════╝    ╚═╝       ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝
                                                                                           


 """)

    # set random seed fopr reprobucibility
    torch.manual_seed(cfg.trainer.random_seed)

    # set ouptut folder
    # Convert the strings to Path objects and join them
    plots_path = Path(cfg.out_path.plots)  # Convert cfg.out_path.plots to a Path object
    method_path = Path(cfg.optimization.method)  # Convert cfg.optimization.method to a Path object

    # Combine the paths
    out_path = plots_path / method_path
    out_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Plots are saved in: {out_path.as_posix()}")

    # device
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if cfg.trainer.use_cuda and torch.cuda.is_available() else "cpu")

    # running problem
    all_traj, all_time = trainer(cfg, device)

    title_map = {
            "nashmtl": "Nash-MTL",
            "cagrad": "CAGrad",
            "pcgrad": "PCGrad",
            "ls": "LS",
            "famo": "FAMO",
            "imtlg": "IMTL-G",
            "graddrop": "GradDrop",
            "alignedmtl": "AlignedMtl",
            "samgs": "SAM-GS",
        }

    
        
    # save computatoin time
    time_path = Path(cfg.out_path.time)
    time_path.mkdir(parents=True, exist_ok=True)
    save_path = time_path /f"{cfg.optimization.method}.time" 
    torch.save(all_time, save_path)

    # Save dictionary as a JSON file
    info_file = {}
    for k,tt in all_traj.items():
        conf = []
        curv = []
        magn = []

        for j in range(cfg.trainer.n_epochs):
            g = torch.from_numpy(tt['grad'][j])
            conf.append(conflict_grads(g))
            curv.append(curvarute_bounding_measure(g))
            magn.append(gradient_magnitude_similarity(g))

        info_file[k] = {"conf" : conf,
                        "curv" : curv,
                        "magn" : magn,          
                        "init" : all_traj[k]['init'],
                        "trjk" : all_traj[k]['traj'],
                        "grad" : all_traj[k]['grad']
                        }

    save_path = out_path / f"{cfg.interest_function.function}_{cfg.problem.scale}_all_traj.json" 
    with open(save_path, 'w') as json_file:
        json.dump(convert_to_serializable(info_file), json_file, indent=4)


        
    # plot pareto
    save_path = out_path / f"{cfg.interest_function.function}_{cfg.problem.scale}_2dPareto.png" 
    ax, fig = plot_2d_pareto(trajectories=all_traj, scale=cfg.problem.scale) #, legend
    ax.set_title(title_map[cfg.optimization.method], fontsize=25)

    fig.savefig(
        save_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    # plot counter
    ax, fig = plot_contour(trajectories=all_traj, scale=cfg.problem.scale,arrwograd=cfg.plotSpecification.arrwograd,method=cfg.optimization.method)
    ax.set_title(title_map[cfg.optimization.method], fontsize=25)

    save_path = out_path /f"{cfg.interest_function.function}_{cfg.problem.scale}_count.png" 
    plt.savefig(
        save_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    # plot 3d
    save_path = plots_path / f"f12_3d.png" 
    if not save_path.exists():
        ax, fig = plot3d(cfg.problem.scale,Interest_function='f12')
        plt.savefig(save_path,bbox_inches="tight",facecolor="white")
        plt.close()

        save_path = plots_path / f"f1_3d.png" 
        ax, fig = plot3d(cfg.problem.scale,Interest_function='f1')
        plt.savefig(save_path,bbox_inches="tight",facecolor="white")
        plt.close()

        save_path = plots_path / f"f2_3d.png" 
        ax, fig = plot3d(cfg.problem.scale,Interest_function='f2')
        plt.savefig(save_path,bbox_inches="tight",facecolor="white")
        plt.close()

    time = {}
    for filename in os.listdir(Path(cfg.out_path.time)):
        if filename.endswith('.time'):
            method = str.replace(filename,'.time','')
            file_path = os.path.join(Path(cfg.out_path.time), filename)
            npzfile = np.load(file_path)
            data_pkl = npzfile[f'{method}/data.pkl']
            data = pickle.loads(data_pkl)
            time[method] = data

    save_path = Path(cfg.out_path.time) / "time.json"
    with open(save_path, 'w') as json_file:
        json.dump(time, json_file)


    logging.info(f"experiment completed")

if __name__ == '__main__':
    main()