import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def d2_dx2(grid, h):
    d2_dx2 = torch.zeros_like(grid)
    d2_dx2[:, 1:-1] = (grid[:, 2:] - 2 * grid[:, 1:-1] + grid[:, :-2]) / (h**2)
    return d2_dx2

def d2_dydx(grid, h):
    d2_dydx = torch.zeros_like(grid)
    d2_dydx[1:-1, 1:-1] = (grid[2:, 2:] - grid[2:, :-2] - grid[:-2, 2:] + grid[:-2, :-2]) / (4 * h**2)
        
def pde_loss(pred, eps, dx):
    mu0 = 1.26663706 * 1e-6 # m kg s-2 A-2
    omega = 2 * np.pi * 2.99792658e8 / ( 1050e-9 * torch.sqrt(eps) ) 
    eps0 = 8.854187817e-12 # F / m
    d2Hy_dx2 = d2_dx2(pred, dx)
    pde_loss = omega[:, :, 2:-2]**2 * mu0 * pred[:, :, 2:-2] + d2Hy_dx2[:, :, 2:-2] / ( eps[:, :, 2:-2] * eps0 )
    return pde_loss
        
        
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        # self.alpha = alpha
        # self.dx = dx

    def forward(self, pred_hr_fields, hr_fields, hr_eps):
        
        # Set up constants and vars
        alpha = 0.1
        dx = 1600e-9 / pred_hr_fields.shape[3]
        
        # mu0 = 1.26663706 * 1e-6 # m kg s-2 A-2
        # omega = 2 * np.pi * 2.99792658e8 / ( 1050e-9 * torch.sqrt(hr_eps) ) 
        # eps0 = 8.854187817e-12 # F / m
        
        real_pred = pred_hr_fields[:, 0, :, :]
        imag_pred = pred_hr_fields[:, 1, :, :]
        real_target = hr_fields[:, 0, :, :]
        imag_target = hr_fields[:, 1, :, :]
                
        # PDE loss
        real_pde_loss = pde_loss(real_pred, hr_eps, dx)
        imag_pde_loss = pde_loss(imag_pred, hr_eps, dx)

        # Take MAE and multiply by 1e-26 to counterbalance large prefactors (i.e. normalize)
        real_pde_loss = torch.mean(torch.abs(real_pde_loss)) * 1e-26
        imag_pde_loss = torch.mean(torch.abs(imag_pde_loss)) * 1e-26
        total_pde_loss = real_pde_loss + imag_pde_loss

        #Scale PDE Loss
        total_loss = alpha * total_pde_loss
        print(f"total_pde_loss: {total_pde_loss}")
        
        # Validate that PDE loss is reasonable for the ground truth
        # Idea: normalize PDE loss by the results for the ground truth     
        real_target_pde_loss = pde_loss(real_target, hr_eps, dx)
        imag_target_pde_loss = pde_loss(imag_target, hr_eps, dx)
        
        real_target_pde_loss = torch.mean(torch.abs(real_target_pde_loss)) * 1e-26
        imag_target_pde_loss = torch.mean(torch.abs(imag_target_pde_loss)) * 1e-26
        
        total_target_pde_loss = real_target_pde_loss + imag_target_pde_loss
        print(f"total_target_pde_loss: {total_target_pde_loss}")
        print(f"pde loss ration (pred/target): {total_pde_loss/total_target_pde_loss}")
        
        return total_loss

