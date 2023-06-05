import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        # self.alpha = alpha
        # self.dx = dx

    def forward(self, pred_hr_fields, hr_fields, hr_eps):
        
        # Set up constants and vars
        alpha = 0.1
        dx = 1600e-9 / pred_hr_fields.shape[3]
        
        mu0 = 1.25663706 * 1e-6 # m kg s-2 A-2
        omega = 2 * np.pi * 2.99792458e8 / ( 1050e-9 ) 
        eps0 = 8.854187817e-12 # F / m
        
        real_pred = pred_hr_fields[:, 0, :, :]
        imag_pred = pred_hr_fields[:, 1, :, :]
        real_target = hr_fields[:, 0, :, :]
        imag_target = hr_fields[:, 1, :, :]
        
        # MSE Loss
        
        real_mse_loss = torch.mean(torch.abs(real_pred - real_target))
        imag_mse_loss = torch.mean(torch.abs(imag_pred - imag_target))
        
        total_mse_loss = real_mse_loss + imag_mse_loss
                
        # PDE loss
        # For NN evaluation of d2Hy_dx2, must first pad input
        # When evaluating PDE loss, do not consider the border in the sum
                
        real_pred_padded = F.pad(real_pred, pad=(1, 1), mode='constant', value=0)
        imag_pred_padded = F.pad(imag_pred, pad=(1, 1), mode='constant', value=0) # double check
        
        real_d2Hy_dx2 = ( real_pred_padded[:, :, 1:-3] - real_pred_padded[:, :, 3:-1] ) / 2
        real_d2Hy_dx2 /= dx**2
        imag_d2Hy_dx2 = ( imag_pred_padded[:, :, 1:-3] - imag_pred_padded[:, :, 3:-1] ) / 2
        imag_d2Hy_dx2 /= dx**2
        
        # print(f"real_pred shape: {real_pred.shape}")
        # print(f"real_pred_padded shape: {real_pred_padded.shape}")
        # print(f"real_d2Hy_dx2 shape: {real_d2Hy_dx2.shape}")
        # print(f"real_pred[:, :, 1:-1] shape: {real_pred[:, :, 1:-1].shape}")
        # print(f"hr_eps[:, :, 1:-1] shape: {hr_eps[:, :, 1:-1].shape}")
        # print(f"real_d2Hy_dx2 shape: {real_d2Hy_dx2.shape}")
        
        real_pde_loss = omega**2 * mu0 * real_pred[:, :, 1:-1] - real_d2Hy_dx2 / ( hr_eps[:, :, 1:-1] * eps0 )
        imag_pde_loss = omega**2 * mu0 * imag_pred[:, :, 1:-1] - imag_d2Hy_dx2 / ( hr_eps[:, :, 1:-1] * eps0 )

        # Take MAE and multiply by 1e-24 to counterbalance large prefactors (i.e. normalize)
        real_pde_loss = torch.mean(torch.abs(real_pde_loss)) * 1e-24
        imag_pde_loss = torch.mean(torch.abs(imag_pde_loss)) * 1e-42
        total_pde_loss = real_pde_loss + imag_pde_loss
        
        # Combine MSE and PDE MAE

        total_loss = total_mse_loss + alpha * total_pde_loss
        print(f"total_mse_loss: {total_mse_loss}")
        print(f"total_pde_loss: {total_pde_loss}")
        print(f"total_loss: {total_loss}")
        
        return total_loss
