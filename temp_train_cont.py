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
    momentum: float = 0.99

    optimizer = optim.SGD(params = model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = nn.MSELoss()

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        batchcount = 1
        print("epoch " + str(epoch) + " started")
        for batch in train_loader:

            optimizer.zero_grad(set_to_none=True)

            print("processing batch " + str(batchcount))
            batchcount+=1
            metagratings, ground_truth = batch[1], batch[0] # batch[0] HR, batch[1] LR, batch[2] point coord Samples, batch[3] point_value
            y_hat = model(metagratings)
            
            loss = loss_fn(y_hat,ground_truth)
            print("loss", loss.item())
            loss.backward()

            optimizer.step()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=2)
    
    train_model(
            model=model,
            epochs=6,
            batch_size=30,
            learning_rate=0.01,
            device=device)
    
    torch.save(model.state_dict(), 'model.pth')
