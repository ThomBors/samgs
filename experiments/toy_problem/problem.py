#!/usr/bin/env python

import torch
from torch import nn

LOWER = 0.000005


class Toy(nn.Module):
    def __init__(self, scale=1.0, scale_both_losses=1.0,Interest_function='f12'):
        super(Toy, self).__init__()
        assert Interest_function in ['f1', 'f2', 'f12'], "Interest_function must be one of ['f1', 'f2', 'f12']"
        self.centers = torch.Tensor([[-3.0, 0], [3.0, 0]])
        self.scale = scale
        self.scale_both_losses = scale_both_losses
        self.Interest_function = Interest_function

    def Function1(self, x1, x2, compute_grad=False):
        f1 = torch.clamp((0.5 * (-x1 - 7) - torch.tanh(-x2)).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2 * 0.5), 0)

        f1_sq = ((-x1 + 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2 * 0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f1 *= self.scale

        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            return f1, g11, g12
        else:
            return f1
        
    def Function2(self, x1,x2, compute_grad=False):
        f2 = torch.clamp((0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2 * 0.5), 0)

        f2_sq = ((-x1 - 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2 * 0.5), 0)

        f2 = f2 * c1 + f2_sq * c2

        if compute_grad:
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            return f2, g21, g22
        else:
            return f2

    def forward(self, x, compute_grad=False):
        x1 = x[0]
        x2 = x[1]

        if self.Interest_function == 'f1':
            f1, g11, g12 = self.Function1(x1,x2,compute_grad)
            f = torch.stack([f1]) 
            if compute_grad:
                g = torch.Tensor([[g11], [g12]])

        elif self.Interest_function == 'f2':
            f2, g21, g22 = self.Function2(x1,x2,compute_grad)
            f = torch.stack([f2]) 
            if compute_grad:
                g = torch.Tensor([[g21], [g22]])

        elif self.Interest_function == 'f12':
            f1, g11, g12 = self.Function1(x1,x2,compute_grad)
            f2, g21, g22 = self.Function2(x1,x2,compute_grad)

            f = torch.stack([f1, f2]) * self.scale_both_losses
            if compute_grad:
                g = torch.Tensor([[g11, g21], [g12, g22]])

       
        if compute_grad:
            return f, g
        else:
            return f

    def batch_forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        if self.Interest_function == 'f1':
            f1 = self.Function1(x1,x2)
            f = torch.cat([f1.view(-1, 1)], -1) * self.scale_both_losses 
            
        elif self.Interest_function == 'f2':
            f2 = self.Function2(x1,x2)
            f = torch.cat([f2.view(-1, 1)], -1) * self.scale_both_losses
            
        elif self.Interest_function == 'f12':
            f1 = self.Function1(x1,x2)
            f2 = self.Function2(x1,x2)
            f = torch.cat([f1.view(-1, 1), f2.view(-1, 1)], -1) * self.scale_both_losses
            
        return f
    