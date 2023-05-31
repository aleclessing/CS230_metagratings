import sys
sys.path.append('../')
import os
import torch
import math
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torch import autograd
import torch.nn as nn
from tqdm import tqdm
import logging
import scipy.io as io
import numpy as np;
from ceviche import *
from ceviche import fdfd_hz;
import time;
Tensor = torch.FloatTensor

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / math.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum

class params:
    wavelength = 1050      # class variable shared by all instances
    dL = 6.25; #e-9
    angle = 10

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance  

def data_generation(full_pattern: np.array, params) -> np.array:
    
    wavelength = params.wavelength*1e-9;
    omega = 2 * np.pi * C_0 / wavelength;
    k_sub = 2 * np.pi / wavelength * 1.45; # Used for cancelling backward field; source embed in substrate with n = 1.45
    dL = params.dL;
    grid_shape = Nx, Ny = int(1600/dL), int(5200/dL);
    npml = [0, 40]; # Periodic in x direction
    eps_r = np.ones(grid_shape);
    # Substrate domain
    eps_r[:, :int(2431.25/dL)] = 2.1025; # Substrate dielectric = 1.45**2
    
    
    # Remeber to rescale the pattern from 0/1 pattern to realistic values !!!!!
    full_pattern = full_pattern*(12.726271412100001 - 1) + 1; # Si polycrystal silicon
#     full_pattern = full_pattern*(2.4803**2 - 1) + 1; #TiO2 anatase 
#     full_pattern = full_pattern*(9.72940864 - 1) + 1; #GaP
    
    # Grating pattern
    eps_r[:, int(2431.25/dL):int(2800/dL)] = full_pattern;
    
    # Set up the FDFD simulation for TM
    F = fdfd_hz(omega, dL*1e-9, eps_r, npml);
    
    # Source
    source_amp = 64e9/dL/dL;
    source_loc_y = int(2320/dL);
    
    # Define the source as just a constant source along x at `y = source_loc_y`, modeling plane wave
    source = np.zeros(grid_shape, dtype=complex);
    source[:, source_loc_y] = source_amp;
    
    # Add a source directly behind to cancel back traveling wave (for TFSF effect)
    source[:, source_loc_y-1] = source[:, source_loc_y] * np.exp(-1j * k_sub * dL*1e-9 - 1j * np.pi);
    
    # Solve the FDFD simulation for the fields, offset the phase such that Ex has 0 phase at the center of the bottom row of the window
    Ex_forward, Ey_forward, Hz_forward = F.solve(source);
    Hz_out_forward = Hz_forward[:, int(2400/dL):int(2800/dL)]*np.exp(-1j*0.784368210509431);
    Ex_out_forward = Ex_forward[:, int(2400/dL):int(2800/dL)]*np.exp(-1j*0.784368210509431);
    Ey_out_forward = Ey_forward[:, int(2400/dL):int(2800/dL)]*np.exp(-1j*0.784368210509431);
    
    return -Hz_out_forward, Ex_out_forward, Ey_out_forward;

if __name__ == '__main__':
    dL=6.25 # grid cell size = 6.25 nm

    ## GENERATE PATTERN ##
    full_pattern = np.zeros([int(1600/dL), int(368.75/dL)]); # image size 1600 nm by 368.75 nm
    # Converting pattern into binary 0/1 in numpy
    pattern = torch.sign(torch.rand(int(1600/dL))*2. -1.);          
    pattern = (pattern + 1)/2;
    pattern = pattern.numpy(); 
    #End of Conversion
    thickness = 58 # 52 * 6.25 nm for metagrating thickness
    full_pattern[:, 0:thickness] = pattern[:, np.newaxis];

    # Near-field computation
    Hy_out_forward, Ex_out_forward, Ez_out_forward = data_generation(full_pattern, params);
