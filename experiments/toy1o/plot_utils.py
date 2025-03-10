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
    n = 500
    xl = 11
    F = Toy(scale=scale)

    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    cmap = colors.LinearSegmentedColormap.from_list("custom_plasma", 
                    [cm.plasma(0.), cm.plasma(0.2), cm.plasma(0.9)])  


    yy = -8.3552

    Yv = Ys.mean(1)
    for i,tt in trajectories.items():
        plt.plot(tt['init'][0],tt['init'][1], marker='o', markersize=10, zorder=5, color='k')
    plt.plot([-7, 7], [yy, yy], linewidth=8.0, zorder=0, color='#FFB14E',label='Pareto Front')
    plt.plot(0, yy, marker='*', markersize=15, zorder=3, color='k',label = 'Global Optimum')


    c = plt.contour(X, Y, Yv.view(n,n), cmap=cmap, linewidths=2.0,levels=7)

    for i,tt in trajectories.items():
        dd = torch.from_numpy(np.array(tt['traj']))
        l = dd.shape[0]
        color_list = np.zeros((l,3))
        color_list[:,0] = 1.
        color_list[:,1] = np.linspace(0, 1, l)
        colors_l = plt.cm.Blues(np.linspace(0.3, 1, dd.shape[0]))
        ax.scatter(dd[:,0], dd[:,1], color=colors_l, s=6, zorder=2)
        ax.scatter(
                dd[-1, 0], 
                dd[-1, 1],
                color="#E60000",
                zorder = 100,
                s=30,
                label=f"Step {l}" if i == 0 else None
            )

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xticks([-7,0, 7])
    ax.set_yticks([-7,0, 7])
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)

    ax.set_xlabel(r"$\theta_1$", fontsize=20,  loc='right')  
    ax.set_ylabel(r"$\theta_2$", fontsize=20,  loc='top') 

    plt.tight_layout()
    return ax, fig


def plot_2d_pareto(trajectories: dict, scale):

    fig, ax = plt.subplots(figsize=(3, 3))

    F = Toy(scale=scale)

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(torch.from_numpy(np.array(res["traj"]))))

    yy = -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).double()

    Ys = F.batch_forward(Xs)

    ax.plot(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        "-",
        linewidth=8,
        color="#FFB14E",
        label="Pareto Front",
    ) 

    Xs = torch.tensor([[ 2.0524e-07, -8.3545e+00]], dtype=torch.float64)
    Ys = F.batch_forward(Xs)

    ax.scatter(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        color="k",
        zorder = 90,
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
            zorder=5 + 2 ,
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
    ax.set_xticks([0,10, 20])
    ax.set_yticks([0,10, 20])
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    ax.set_xlabel(r"$\mathcal{L}_1$", fontsize=20,  loc='right')  
    ax.set_ylabel(r"$\mathcal{L}_2$", fontsize=20,  loc='top')  
    
    return ax, fig


def plot3d(scale,n = 500,xl=11,Interest_function='f12'):

    F = Toy(scale=scale,Interest_function=Interest_function)

    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=plt.cm.viridis)
    # print(Ys.mean(1).min(), Ys.mean(1).max())

    ax.set_zticks([-16, -8, 0, 8])
    ax.set_zlim(-20, 10)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label1.set_fontsize(15)

    ax.view_init(25)
    plt.tight_layout()
    return ax, fig