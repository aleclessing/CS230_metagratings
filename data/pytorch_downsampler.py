import numpy as np
import torch
import torch.nn.functional as F

test_data = np.load('test_ds.npz')
train_data = np.load('train_ds.npz')

downsamp = 2

hr_nx = test_data['Hy_fields'][0][0].shape[0]
hr_nz = test_data['Hy_fields'][0][0].shape[1]
test_num_examples = test_data['Hy_fields'].shape[0]

train_num_examples = train_data['Hy_fields'].shape[0]

num_examples = train_num_examples + test_num_examples

lr_nx = int(hr_nx/downsamp)
lr_nz = int(hr_nz/downsamp)


fields = ['Re_Hy', 'Im_Hy', 'eps']

hr_data = np.empty((num_examples, len(fields), hr_nx, hr_nz))

hr_data[:test_num_examples, 1, :, :] = test_data['Hy_fields'][:, 0, :, :]
hr_data[:test_num_examples, 2, :, :] = test_data['Hy_fields'][:, 1, :, :]
hr_data[:test_num_examples, 0, :, :]= test_data['structures'][:, 0, :, :]

hr_data[test_num_examples:, 1, :, :] = train_data['Hy_fields'][:, 0, :, :]
hr_data[test_num_examples:, 2, :, :] = train_data['Hy_fields'][:, 1, :, :]
hr_data[test_num_examples:, 0, :, :]= train_data['structures'][:, 0, :, :]

print(hr_data.shape)

hr_data = torch.from_numpy(hr_data)

weights = torch.ones((3, 1, downsamp, downsamp), dtype=torch.double)/(downsamp**2)

lr_data = F.conv2d(hr_data, weight=weights, stride=2, groups=3)

np.save('lr_data', lr_data)
np.save('hr_data', hr_data)