import argparse
import jnet
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import dataloader as loader
from jnet import JNet
from unet import UNet

def train_model(model, epochs, batch_size, learning_rate, device):
    
    # 1. Open Dataset
    dataset = loader.MetaGratingDataLoader(return_hres=True, hr_data_filename='data/hr_data.npz', lr_data_filename='data/lr_data.npz')
    # print(dataset[0]) # First sample out of 100 with information on the Re, Im, and eps 
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1) # 90-10 split
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    weight_decay: float = 1e-8
    momentum: float = 0.999

    optimizer = optim.SGD( params = model.parameters (), lr=0.1, momentum=0.8 )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        batchcount = 1
        print("epoch " + str(epoch) + " started")
        for batch in train_loader:
            print("processing batch " + str(batchcount))
            batchcount+=1
            metagratings, ground_truth = batch[1], batch[0] # batch[0] HR, batch[1] LR, batch[2] point coord Samples, batch[3] point_value
            print('c0', metagratings.shape)
            y_hat = model(metagratings)
            loss = nn.MSELoss(y_hat,ground_truth)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on metagrating dataset')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=2)

    train_model(
            model=model,
            epochs=2,
            batch_size=5,
            learning_rate=0.001,
            device=device)
