import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from unet import UNet
from utils.utils import plot_hr_lr_sr
import jnet

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
    lr_img = lr_img.unsqueeze(0) # Add a batch dimension
    lr_img = lr_img.to(device=device, dtype=torch.float32)

    with torch.no_grad(): # sets up context where no gradient computation is performed
        output = net(lr_img).cpu() # passes img thru the nn model to obtain output logits / probabilities
    
    return output.long().squeeze(0).numpy() 


def get_args():
    parser = argparse.ArgumentParser(description='Predict HR Hy field from LR Hy field')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--truth', '-t', metavar='TRUTH', nargs='+', help='Filenames of ground truth fld array')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input fld array', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output fld array')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_SR_OUT.npy'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    truth_files = args.truth
    in_files = args.input
    out_files = get_output_filenames(args)

    # net = UNet(in_channels=3)
    net = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        # must first open a data file and read it into a numpy array
        lr_img = np.load(filename)
        sr_img = predict_img(net=net, lr_img=lr_img, device=device)

        if not args.no_save:
            out_filename = out_files[i]
            np.save(out_filename, sr_img)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            hr_img = np.load(truth_files[i])
            plot_hr_lr_sr(hr_img, lr_img, sr_img)