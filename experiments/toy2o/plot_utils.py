#!/usr/bin/env python

import torch
import numpy as np
from scipy.optimize import minimize_scalar

from matplotlib import pyplot as plt
from matplotlib import path as mpath
from matplotlib import cm
from matplotlib import colors

import seaborn as sns
from PIL import Image

import logging
from tqdm import tqdm

from methods.utils import gradient_aggegation
from problem import Toy

def plot_contour(trajectories: dict, scale,arrwograd=1,method='ls'): 
    n = 200
    xl = 11
    F = Toy(scale=scale)

    fig,ax = plt.figure(figsize=(3, 3))
    x = np.linspace(-8, 8, n)
    y = np.linspace(-12.5, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    cmap = colors.LinearSegmentedColormap.from_list(
        "custom_plasma", [cm.plasma(0.), cm.plasma(0.2), cm.plasma(0.9)]
    )

    yy = -10.84

    # Initial point
    Yv = Ys.mean(1)
    for i, tt in trajectories.items():
        ax.plot(tt['init'][0], tt['init'][1], marker='o', markersize=10, zorder=5, color='k')

    # Optimal solution
    ax.plot([-5.45, 5.45], [yy, yy], linewidth=8.0, zorder=0, color='#FFB14E', label='Pareto Front')
    ax.plot([-5.45, 5.45], [yy, yy], marker='*', markersize=15, zorder=3, color='k', label='Global Optimum')

    # Contour plot
    contour = ax.contour(X, Y, Yv.view(n, n), cmap=cmap, linewidths=2.0, levels=7)

    # Trajectory
    for i, tt in trajectories.items():
        dd = torch.from_numpy(np.array(tt['traj']))
        colors_ = cm.Blues(np.linspace(0.3, 1, dd.shape[0]))

        ax.scatter(dd[:, 0], dd[:, 1], color=colors_, s=6, zorder=2)
        ax.scatter(
            dd[-1, 0], 
            dd[-1, 1],
            color="#E60000",
            zorder=100,
            s=30,
            label=f"Step {dd.shape[0]}" if i == 0 else None
        )

    # Axis properties
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    ax.set_xlim(-8, 8)
    ax.set_ylim(-12.5, xl)

    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)

    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$\theta_1$", fontsize=20, loc='right')
    ax.set_ylabel(r"$\theta_2$", fontsize=20, loc='top')

    # Ensure aspect ratio
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    return ax, fig

def plot_2d_pareto(trajectories: dict, scale,pareto):
    fig, ax = plt.subplots(figsize=(5, 5))

    F = Toy(scale=scale)

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(torch.from_numpy(res["traj"])))


    ax.plot(
        pareto[:, 0],
        pareto[:, 1],
        "-",
        linewidth=8,
        color="#FFB14E",
        label="Pareto Front",
    )  # Pareto front

    ax.scatter(
        [-51,-23.7],
        [-23.7,-51],
        color="k",
        zorder=90,
        marker="*",
        s=200,
        label="Global Optimum"
    )

    for i, tt in enumerate(losses):

        ax.scatter(
            tt[0, 0],
            tt[0, 1],
            color="k",
            marker='o',
            edgecolors='#ffffff',
            linewidths=0.8,
            s=100,
            zorder=5 + 2,
            label="Initial Point" if i == 0 else None,
        )

        colors = plt.cm.Blues(np.linspace(0.3, 1, tt[:-1].shape[0]))
        ax.scatter(tt[:-1, 0], 
                tt[:-1, 1], 
                color=colors, 
                s=5, 
                zorder=4
                )
        
        ax.scatter(
            tt[-1, 0], 
            tt[-1, 1],
            color="#E60000",
            zorder = 100,
            s=30,
            label=f"Step {tt.shape[0]}" if i == 0 else None
        )

    sns.despine()

    ax.set_xticks([-45,-25])
    ax.set_yticks([-45,-25])
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r"$\mathcal{L}_1$", fontsize=20,  loc='right')  
    ax.set_ylabel(r"$\mathcal{L}_2$", fontsize=20,  loc='top') 
    return ax, fig


def plot3d(scale,n = 500,xl=11,Interest_function='f12'):

    F = Toy(scale=scale,Interest_function=Interest_function)

    X, Y = create_grid(Interest_function=Interest_function,n=n)

    new_cmap = colors.LinearSegmentedColormap.from_list("custom_plasma", 
                    [cm.plasma(0.), cm.plasma(0.2), cm.plasma(0.9)])   

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=new_cmap)

    ax.set_zticks([-20,-10, 0, 10])
    ax.set_zlim(-30, 5)

    ax.set_xticks([-5,0, 5])
    ax.set_yticks([-5, 0, 5])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label1.set_fontsize(15)

    ax.view_init(25)
    ax.text(0, -15, -35, r"$\theta_1$", fontsize=25, rotation=0, color="black")  
    ax.text(7, 10, -55, r"$\theta_2$", fontsize=25, color="black")
    ax.set_title(r"$\mathcal{L}_{MTL}$", fontsize=30)
    plt.tight_layout()
    return ax, fig

def create_grid(Interest_function,n=200):
    if Interest_function == 'f12':
        x = np.linspace(-8, 8, n)
        y = np.linspace(-8, 8, n)
    elif Interest_function == 'f1':
        x = np.linspace(-8, 18.5, n)
        y = np.linspace(-8, 11.8, n)
    else:
        x = np.linspace(-18.5, 8, n)
        y = np.linspace(-8, 11.8, n)
    return np.meshgrid(x, y)


def pareto_front_toy(scale,function_name='f12', n=1000, xlim=(-19, 19), zlim=(-13, 5)):
    F = Toy(scale=scale,Interest_function=function_name)
    # Generate the grid of x and z values
    x = np.linspace(xlim[0], xlim[1], n)
    z = np.linspace(zlim[0], zlim[1], n)
    
    # Create a meshgrid
    X, Z = np.meshgrid(x, z)
    
    # Flatten the grid and stack the x and y coordinates
    XZs = torch.Tensor(np.transpose(np.array([X.flat, Z.flat]))).float()
    print('batch_forward')
    # Perform batch forward pass to compute y1 and y2
    Ys = F.batch_forward(XZs).numpy()  # Convert to numpy for processing
    
    # Ys[:, 0] corresponds to y1 and Ys[:, 1] corresponds to y2
    points = [(Ys[i, 0], Ys[i, 1], XZs[i, 0].item(), XZs[i, 1].item()) for i in range(len(XZs))]
    
    # Now we need to find non-dominated points (Pareto front)
    pareto_points = []
    for i, (y1_i, y2_i, x_i, z_i) in enumerate(tqdm(points)):
        dominated = False
        for j, (y1_j, y2_j, x_j, z_j) in enumerate(points):
            if i != j and y1_j <= y1_i and y2_j <= y2_i and (y1_j < y1_i or y2_j < y2_i):
                dominated = True
                break
        if not dominated:
            pareto_points.append((y1_i, y2_i, x_i, z_i))
    
    return torch.tensor(pareto_points)