import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
# from torchvision import transforms
from unet import UNet
import jnet
import dataloader


import matplotlib.pyplot as plt

def plot_hr_lr_sr(hr_data, lr_data, sr_data, int_data=None):
    """
    Plots the HR ground truth, LR image, and HR super-resolved (SR) image 
    side-by-side for both the real and imaginary parts of the Hy field.
    Want two columns (real and imaginary) and three rows (HR, LR, and SR).

    Args:
        lr_data: the LR array for single example; shape [3, 32, 128] np array
        hr_data: the HR array for single example; shape [2, 64, 256] np array
        Note: the 3 channels are permitivities and the Re and Im parts of the 
        Hy field.
    """
    if int_data is None:
        num_rows = 3
    else:
        num_rows = 4

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 10))
    fig.suptitle('HR, LR, and SR Hy field')
    axs[0, 0].imshow(hr_data[0, :, :])
    axs[0, 0].set_title('HR Re(Hy)')
    axs[0, 1].imshow(hr_data[1, :, :])
    axs[0, 1].set_title('HR Im(Hy)')
    axs[1, 0].imshow(lr_data[0, :, :])
    axs[1, 0].set_title('LR Re(Hy)')
    axs[1, 1].imshow(lr_data[1, :, :])
    axs[1, 1].set_title('LR Im(Hy)')
    axs[2, 0].imshow(sr_data[0, :, :])
    axs[2, 0].set_title('SR Re(Hy)')
    axs[2, 1].imshow(sr_data[1, :, :])
    axs[2, 1].set_title('SR Im(Hy)')
    if int_data is not None:
        axs[3, 0].imshow(int_data[0, :, :])
        axs[3, 0].set_title('Int Re(Hy)')
        axs[3, 1].imshow(int_data[1, :, :])
        axs[3, 1].set_title('Int Im(Hy)')

    plt.savefig('pred.png')
    plt.show()


# I will assume that the lr input image is a 3D array [3, 32, 128]
# and the hr truth is a 3D array [2, 64, 256] (permitivities have been removed).
# For the time being, we will not do any preprocessing of the input image

def predict_img(net, lr_img, hr_eps, device):
    """
    Predicts the output image given the input image and the network model.

    Args:
        net: the network model
        lr_img: the input image assumed to be a 3D array [3, 32, 128]
        device: the device to run the model on (CPU or GPU)

    Returns:
        the output image as a numpy array of long ints with shape [32, 128]
    """
    net.eval()
    lr_img = lr_img
    lr_img = torch.tensor(lr_img).unsqueeze(0)
    hr_eps = torch.tensor(hr_eps).unsqueeze(0)

    print(hr_eps.shape, lr_img.shape)

    with torch.no_grad(): # sets up context where no gradient computation is performed
        output = net(lr_img, hr_eps).cpu()
    
    return output.squeeze(0).numpy() 


def get_args():
    parser = argparse.ArgumentParser(description='Predict HR Hy field from LR Hy field')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--truth', '-t', metavar='TRUTH', nargs='+', help='Filenames of ground truth fld array')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input fld array', required=True)
    parser.add_argument('--exnum', '-e', metavar='EXNUM', nargs='+', help='Example number of interest, giving lr and hr data', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output fld array')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed', default=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks', default=False)
    
    return parser.parse_args()

"""
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_SR_OUT.npy'

    return args.output or list(map(_generate_name, args.input))
"""

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # truth_files = args.truth
    # in_files = args.input

    hr_eps, lr_fields, hr_fields = dataloader.MetaGratingDataLoader(return_hres=True)[int(args.exnum[0])]

    #out_files = get_output_filenames(args)

    net = jnet.TCAJNet( static_channels=1, dynamic_channels=2, upsampling_layers=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    #logging.info(f'Using device {device}')

    net = torch.load(args.model)

    # must first open a data file and read it into a numpy array
    sr_fields = predict_img(net=net, lr_img=lr_fields, hr_eps=hr_eps, device=device)

    if not args.no_save:
        pass
        #out_filename = out_files[i]
        #np.save(out_filename, sr_img)
        #logging.info(f'Mask saved to {out_filename}')

    if args.viz:
        #logging.info(f'Visualizing results for image {filename}, close to continue...')
        plot_hr_lr_sr(hr_fields, lr_fields, sr_fields)
