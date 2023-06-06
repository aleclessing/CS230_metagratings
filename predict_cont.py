import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import cont_jnet
import dataloader


def predict_fields(net, lr_fields, hr_eps, pt_coos):
    net.eval()

    lr_fields = torch.tensor(lr_fields).unsqueeze(0)
    hr_eps = torch.tensor(hr_eps).unsqueeze(0)
    pt_coos = torch.tensor(pt_coos).unsqueeze(0)

    print(lr_fields.shape, hr_eps.shape, pt_coos.shape)
    sr_fields = net(lr_fields, hr_eps, pt_coos)

    print(sr_fields.shape)

    return sr_fields


def diagnose(pt_vals, pt_coos, sr_vals):
    print(np.sqrt(np.mean((pt_vals.detach().numpy()-sr_pt_fields.detach().numpy())**2)))


def predict_plot(lr_fields, hr_fields, pt_coos, pt_vals, sr_vals):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    plt.imshow(hr_fields[0])

    cbar = plt.colorbar()
    color_func = lambda num: cbar.cmap((num - cbar.vmin)/(cbar.vmax - cbar.vmin))

    xs = ((1+pt_coos[:, 0])/2)*hr_fields[0].shape[0]
    zs = ((1+pt_coos[:, 1])/2)*hr_fields[0].shape[1]
    
    axs[1].scatter(x=zs, y=xs, c=color_func(sr_vals[:,0]), s=100, edgecolors='black')

    print("jevoiej", pt_vals.shape, sr_vals.shape)

    axs[0].scatter(pt_vals[:,0], sr_vals[:,0])

    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Predict HR Hy field from LR Hy field')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--exnum', '-e', metavar='EXNUM', nargs='+', help='Example number of interest, giving lr and hr data', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output fld array')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed', default=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks', default=False)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    scale_factor = 8

    hr_eps, lr_fields, pt_coos, pt_vals, hr_fields = dataloader.MetaGratingDataLoader(return_hres=True, n_samp_pts=20, lr_data_filename='' )[int(args.exnum[0])]

    net = torch.load(args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # must first open a data file and read it into a numpy array
    sr_pt_fields = predict_fields(net, lr_fields, hr_eps, pt_coos)

    pt_vals = torch.tensor(pt_vals)
    pt_vals = pt_vals.unsqueeze(0)
    print("pt_vals", pt_vals.shape)

    diagnose(pt_vals, pt_coos, sr_pt_fields)

    if args.viz:
        predict_plot(lr_fields, hr_fields, pt_coos, pt_vals.detach().numpy()[0], sr_pt_fields.detach().numpy()[0])
