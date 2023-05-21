import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from unet import UNet
from utils import plot_hr_lr_sr
import jnet
import dataloader

# I will assume that the lr input image is a 3D array [3, 32, 128]
# and the hr truth is a 3D array [2, 64, 256] (permitivities have been removed).
# For the time being, we will not do any preprocessing of the input image

def predict_img(net, lr_img, device):
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

    with torch.no_grad(): # sets up context where no gradient computation is performed
        output = net(lr_img).cpu() # passes img thru the nn model to obtain output logits / probabilities
    
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


    hr_img, lr_img = dataloader.MetaGratingDataLoader(return_hres=True, n_samp_pts=0)[int(args.exnum[0])]

    #out_files = get_output_filenames(args)

    net = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    #logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    #logging.info('Model loaded!')

    # must first open a data file and read it into a numpy array
    sr_img = predict_img(net=net, lr_img=lr_img, device=device)

    if not args.no_save:
        pass
        #out_filename = out_files[i]
        #np.save(out_filename, sr_img)
        #logging.info(f'Mask saved to {out_filename}')

    if args.viz:
        #logging.info(f'Visualizing results for image {filename}, close to continue...')
        plot_hr_lr_sr(hr_img, lr_img, sr_img)