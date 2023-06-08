
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import warnings


class MetaGratingDataLoader(Dataset):
    """Pytorch Dataset instance for loading Metagratings 2D dataset
    Loads 
    """

    def __init__(self, hr_data_filename='data/metanet_hr_data.npy', lr_data_filename='data/metanet_lr_data_downsamp8.npy', n_samp_pts=0, return_hres=False):
        
        self.hr_data_filename = hr_data_filename
        self.lr_data_filename = lr_data_filename
        self.n_samp_pts = n_samp_pts

        self.hr_data = np.load(hr_data_filename, mmap_mode='r')

        self.lr_data = np.load(lr_data_filename, mmap_mode='r')

        self.n_samples, self.nc, self.nx_hr, self.nz_hr = self.hr_data.shape

        self.return_hres = return_hres

        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Get the random cutout data cube corresponding to idx.

        Args:
          idx: int, index of the example to return. must be smaller than len(self).

        Returns:
          Set of pytorch tensors that consistute one example
        """

        lres_space = np.array(self.lr_data[idx])
        hres_space = np.array(self.hr_data[idx])

        return_tensors = [hres_space[0], lres_space[1:]] #always return hres epsilon grid

        if self.n_samp_pts != 0:
            x_grid_pts = 2*(np.arange(self.nx_hr) + 0.5)/self.nx_hr-1
            z_grid_pts = 2*(np.arange(self.nz_hr) + 0.5)/self.nz_hr-1

            interp = RegularGridInterpolator((x_grid_pts, z_grid_pts), values = hres_space.transpose(1, 2, 0), bounds_error=False, fill_value=None)

            #points range from -1 to +1 in line with convention of pytorch grid_sample
            point_coord = 2*np.random.rand(self.n_samp_pts, 2) - 1 

            point_value = interp(point_coord)
            point_coord = point_coord

            return_tensors = return_tensors + [point_coord, point_value[:,1:]]

        if self.return_hres:
            return_tensors = return_tensors + [hres_space[1:]] #remove eps channel from hres_space

        # cast everything to float32
        return_tensors = [t.astype(np.float32) for t in return_tensors]

        return tuple(return_tensors)


if __name__ == '__main__':
    pass
