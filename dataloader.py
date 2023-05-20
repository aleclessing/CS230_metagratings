
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import warnings

def stack_channels(npdata):
    return np.stack([npdata["eps"], npdata["Re_Hy"], npdata["Im_Hy"]], axis=0)

class MetaGratingDataLoader(Dataset):
    """Pytorch Dataset instance for loading Metagratings 2D dataset
    Loads 
    """

    def __init__(self, hr_data_filename='data/hr_data.npz', lr_data_filename= 'data/lr_data.npz', n_samp_pts=0, return_hres=False):
        
        self.hr_data_filename = hr_data_filename
        self.lr_data_filename = lr_data_filename
        self.n_samp_pts = n_samp_pts

        npdata_hr = np.load(hr_data_filename)
        self.hr_data = stack_channels(npdata_hr)

        npdata_lr = np.load(lr_data_filename)
        self.lr_data = stack_channels(npdata_lr)

        self.nc, self.n_samples, self.nx_hr, self.nz_hr = self.hr_data.shape

        self.scale_hres = np.array([self.nx_hr, self.nz_hr])
        self.return_hres = return_hres

    def __len__(self):
        return 10 #self.hr_data.shape[1]

    def __getitem__(self, idx):
        """Get the random cutout data cube corresponding to idx.

        Args:
          idx: int, index of the example to return. must be smaller than len(self).

        Returns:
          space_crop_hres (*optional): array of shape [4, nz_hres, nx_hres],
          where 4 are the phys channels pbuw.
          space_crop_lres: array of shape [4, nt_lres, nz_lres, nx_lres], where 4 are the phys
          channels pbuw.
          point_coord: array of shape [n_samp_pts_per_crop, 2], where 3 are the x, z dims.
                       CAUTION - point_coord are normalized to (0, 1) for the relative window.
          point_value: array of shape [n_samp_pts_per_crop, 4], where 4 are the phys channels pbuw.
        """
        hres_space = self.hr_data[:, idx, :, :] # [c, example_num, x, z]
        lres_space = self.lr_data[:, idx, :, :]

        return_tensors = [lres_space]

        if self.n_samp_pts != 0:
            interp = RegularGridInterpolator(
                (np.arange(self.nx_hr), np.arange(self.nz_hr)), values = hres_space.transpose(1, 2, 0))

            point_coord = np.random.rand(self.n_samp_pts, 2) * (self.scale_hres - 1)
            point_value = interp(point_coord)
            point_coord = point_coord/(self.scale_hres - 1)

            return_tensors = return_tensors + [point_coord, point_value]

        if self.return_hres:
            return_tensors = [hres_space] + return_tensors

        # cast everything to float32
        return_tensors = [t.astype(np.float32) for t in return_tensors]


        return tuple(return_tensors)


if __name__ == '__main__':
    pass
