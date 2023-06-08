import math
import numpy as np;
from ceviche import fdfd_hz;
from make_eps_grid import gen_eps_grid

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / math.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum

class params:
    wavelength = 1050      # class variable shared by all instances
    dL = 6.25; #e-9
    angle = 10

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance  

def data_generation(eps_grid: np.array, params) -> np.array:
    
    wavelength = params.wavelength*1e-9;
    omega = 2 * np.pi * C_0 / wavelength;
    k_sub = 2 * np.pi / wavelength * 1.45; # Used for cancelling backward field; source embed in substrate with n = 1.45
    dL = params.dL
    grid_shape = Nx, Ny = int(1600/dL), int(5200/dL);
    dL = params.dL;
    eps_r = np.ones(grid_shape);
    npml = [0, 40]; # Periodic in x direction

    #Put our epsilon grid into the appropriate region of the epsilon grid
    eps_r[:eps_grid.shape[0], :eps_grid.shape[1]] = eps_grid
    
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

    eps_grid = gen_eps_grid()

    # Near-field computation
    Hy_out_forward, Ex_out_forward, Ez_out_forward = data_generation(eps_grid, params);

    print(Hy_out_forward.shape)

    import matplotlib.pyplot as plt
    plt.imshow(eps_grid)
    plt.show()

    plt.imshow(Hy_out_forward.real)
    plt.show()
