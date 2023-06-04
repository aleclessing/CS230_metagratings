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
from cont_jnet import ContJNet

def train_model(model, epochs, batch_size, learning_rate, device):
    
    # 1. Open Dataset
    dataset = loader.MetaGratingDataLoader(n_samp_pts=1024)
    
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

    optimizer = optim.AdamW(params = model.parameters(), lr=learning_rate, weight_decay=0.01)
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
            
            lr_grid = batch[0]
            pt_coords = batch[1]
            pt_vals = batch[2]

            pred_pt_vals = model(lr_grid, pt_coords)
            
            loss = loss_fn(pred_pt_vals, pt_vals)
            print("loss", loss.item())
            loss.backward()

            optimizer.step()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ContJNet(static_channels=1, dynamic_channels=2)
    
    train_model(
            model=model,
            epochs=1,
            batch_size=100,
            learning_rate=0.001,
            device=device)
    
    torch.save(model.state_dict(), 'cont_model.pth')
