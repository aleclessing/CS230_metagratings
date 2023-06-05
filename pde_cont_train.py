import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import dataloader as loader
import cont_jnet
from predict_cont import predict_plot

import pde_loss

def gen_coord_grid(shape):
    x_grid_pts = 2*(np.arange(shape[0]) + 0.5)/shape[0]-1
    z_grid_pts = 2*(np.arange(shape[1]) + 0.5)/shape[1]-1

    xs, zs = np.meshgrid(x_grid_pts, z_grid_pts)

    coord_grid = np.stack(xs, zs, axis=2)

    return torch.tensor(coord_grid)

def gen_grid_pt_coords(shape):
    x_grid_pts = 2*(np.arange(shape[0]) + 0.5)/shape[0]-1
    z_grid_pts = 2*(np.arange(shape[1]) + 0.5)/shape[1]-1

    xs, zs = np.meshgrid(x_grid_pts, z_grid_pts)

    print(zs.shape)
    grid_pt_coos = np.stack([xs, zs], axis=2)

    grid_pt_coos = torch.tensor(grid_pt_coos)
    grid_pt_coos = torch.reshape(grid_pt_coos, (shape[0]*shape[1], 2))

    return grid_pt_coos.float()

    


def train_model(model, epochs, batch_size, learning_rate, device , train_writer=None, val_writer=None, model_name='model.pth', txt_log=None, scale_factor=2, weight_decay=0.01, gamma=0.9, save_every_batch=False, prefetch_factor=1, num_cpus=None):
    
    # 1. Open Dataset
    dataset = loader.MetaGratingDataLoader(return_hres=True, lr_data_filename= 'data/metanet_lr_data_downsamp' + str(scale_factor) + '.npy')
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1) # 90-10 split
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, pin_memory=False, multiprocessing_context="fork")

    if num_cpus == None:
        loader_args['num_workers'] = os.cpu_count()
    else:
        loader_args['num_workers'] = num_cpus
    
    if prefetch_factor != None:
        loader_args['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(params = model.parameters(), lr=learning_rate, eps=1e-9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn = pde_loss.CustomLoss()

    global_step = 0

    grid_pt_coos = gen_grid_pt_coords((64, 256))

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        train_batch_loss = []
        train_batchcount = 1
        print("epoch " + str(epoch) + " started")
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            print("processing training batch " + str(train_batchcount))
            
            hr_eps, lr_fields, hr_fields = batch

            grid_pt_coo_batch = grid_pt_coos.unsqueeze(0)
            grid_pt_coo_batch = grid_pt_coo_batch.expand(batch_size, 256*64, 2)
            print(grid_pt_coo_batch.shape)
            sr_grid_fields = model(lr_fields, hr_eps, grid_pt_coo_batch)
            print(sr_grid_fields.shape)
            sr_grid_fields = torch.reshape(sr_grid_fields, (batch_size, 64, 256, 2))
            train_loss = loss_fn(sr_grid_fields, hr_fields, hr_eps)

            print("training loss", train_loss.item())
            train_loss.backward()

            optimizer.step()

            train_batch_loss.append(train_loss.item())
            if txt_log != None:
                print('train ep ' + str(epoch) + ' batch ' + str(train_batchcount) + ' loss ' + str(train_loss.item()), file=txt_log, flush=True)
                #predict_plot(lr_fields[0].detach().numpy(), hr_fields[0].detach().numpy(), pt_coos[0].detach().numpy(), pt_vals[0].detach().numpy(), sr_pt_fields[0].detach().numpy())

            train_batchcount+=1
            if save_every_batch:
                torch.save(model, model_name) 

        scheduler.step()
        avg_train_loss = sum(train_batch_loss) / train_batchcount

        # Evaluate the model on the validation set
        model.eval() # sets the model to evaluation mode
        with torch.no_grad():
            val_batch_loss = []
            val_batchcount = 1
            for batch in val_loader:
                print("processing val batch " + str(val_batchcount))
                
                hr_eps, lr_fields, pt_coos, pt_vals, hr_fields = batch

                sr_pt_fields = model(lr_fields, hr_eps, pt_coos)
                val_loss = loss_fn(sr_pt_fields, pt_vals)

                print("val loss", val_loss.item())
                val_batch_loss.append(val_loss.item())

            avg_val_loss = sum(val_batch_loss) / val_batchcount

            if txt_log != None:
                print('val ep ' + str(epoch) + ' batch ' + str(train_batchcount) + ' loss ' + str(val_loss.item()), file=txt_log, flush=True)

            val_batchcount+=1

        if not save_every_batch:
            torch.save(model, model_name) 

def get_args():
    parser = argparse.ArgumentParser(description='Running training on the Block JNet')
    parser.add_argument('--run_name', '-r', metavar='RUNNAME', nargs='+', help='Run Name', required=True)
    parser.add_argument('--epochs', '-e', metavar='EPS', nargs='+', help='Number of epochs to train', required=True)
    parser.add_argument('--batch', '-b', metavar='BATCHSIZE', nargs='+', help='Size of batch', required=True)
    parser.add_argument('--lr', '-l', metavar='LR', nargs='+', help='Learning Rate', required=True)
    parser.add_argument('--scaling_factor', '-s', metavar='SCALEFACTOR', nargs='+', help='Scaling factor (amount of upsampling to do)', required=True)
    parser.add_argument('--weight_decay', '-w', metavar='WEIGHTDECAY', nargs='+', help='Weight Decay', required=True)
    parser.add_argument('--gamma', '-g', metavar='GAMMA', nargs='+', help='Learning Rate Decay Rate', required=True)
    parser.add_argument('--continue_training', '-c', action='store_true', help='Continue Training Model', default=False)
    parser.add_argument('--save_every_batch', '-eb', action='store_true', help='Save the model at every batch of training (useful for work tweaking the training script)', default=False)
    parser.add_argument('--num_cpus', '-cpu', metavar='CPUNUM', nargs='+', help='Number of CPUs to use', required=False)

    
    return parser.parse_args()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using Device:", device)
    
    # Get hyperparameters
    args = get_args()
    epochs=int(args.epochs[0])
    batch_size=int(args.batch[0])
    learning_rate=float(args.lr[0])
    run_name = args.run_name[0]
    scale_factor = int(args.scaling_factor[0])
    weight_decay = float(args.weight_decay[0])
    gamma = float(args.gamma[0])
    cont = args.continue_training
    if args.num_cpus != None:
        num_cpus = int(args.num_cpus[0])
    else:
        num_cpus = None

    #Create Log File
    lf = open('logs/txt_logs/'+str(run_name)+'_log.txt', 'w+')

    model_name = 'models/model_' + run_name + '.pth'
    #Creat Model
    if cont:
        print("continuing model training based on run name")
        model = torch.load(model_name)
    else:
        print("initializing new model")
        model = cont_jnet.ContJNet(upsampling_layers=int(np.log2(scale_factor)))
    
    train_model(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            model_name=model_name,
            txt_log=lf,
            scale_factor=scale_factor,
            save_every_batch=args.save_every_batch,
            num_cpus=num_cpus)
