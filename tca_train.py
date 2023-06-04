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
import jnet

from torch.utils.tensorboard import SummaryWriter

def train_model(model, epochs, batch_size, learning_rate, device , train_writer, val_writer, model_name='model.pth', txt_log=None, scale_factor=2):
    
    # 1. Open Dataset
    dataset = loader.MetaGratingDataLoader(return_hres=True, lr_data_filename= 'data/metanet_lr_data_downsamp' + str(scale_factor) + '.npy')
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1) # 90-10 split
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, pin_memory=False, multiprocessing_context="fork")
    if device != 'cpu':
        loader_args['num_workers'] = os.cpu_count()
        loader_args['prefetch_factor'] = 2

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    weight_decay: float = 1e-8

    optimizer = optim.AdamW(params = model.parameters(), lr=learning_rate, eps=1e-9, weight_decay=.01)
    loss_fn = nn.MSELoss()

    global_step = 0

    # 5. Begin training
    train_loss_values = [] # For saving epoch loss
    for epoch in range(1, epochs + 1):
        model.train()
        train_batch_loss = []
        train_batchcount = 1
        print("epoch " + str(epoch) + " started")
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            print("processing training batch " + str(train_batchcount))
            
            metagratings, ground_truth = batch[1], batch[0] # batch[0] HR, batch[1] LR, batch[2] point coord Samples, batch[3] point_value
            y_hat = model(metagratings)
            train_loss = loss_fn(y_hat,ground_truth)
            train_writer.add_scalar("Loss", train_loss, epoch)
            print("training loss", train_loss.item())
            train_loss.backward()
            optimizer.step()
            train_batch_loss.append(train_loss.item())

            if txt_log != None:
                print('train ep ' + str(epoch) + ' batch ' + str(train_batchcount) + ' loss ' + str(train_loss.item()), file=txt_log, flush=True)

            train_batchcount+=1

        avg_train_loss = sum(train_batch_loss) / train_batchcount
        train_loss_values.append(avg_train_loss)
        train_writer.add_scalar("Avg Loss", avg_train_loss, epoch)

        # Evaluate the model on the validation set
        model.eval() # sets the model to evaluation mode
        with torch.no_grad():
            val_batch_loss = []
            val_batchcount = 1
            for batch in val_loader:
                print("processing val batch " + str(val_batchcount))
                
                metagratings, ground_truth = batch[1], batch[0]
                y_hat = model(metagratings)
                val_loss = loss_fn(y_hat, ground_truth) 
                val_writer.add_scalar("Loss", val_loss, epoch)
                print("val loss", val_loss.item())
                val_batch_loss.append(val_loss.item())

            avg_val_loss = sum(val_batch_loss) / val_batchcount
            val_writer.add_scalar("Avg Loss", avg_val_loss, epoch)

            if txt_log != None:
                print('val ep ' + str(epoch) + ' batch ' + str(train_batchcount) + ' loss ' + str(val_loss.item()), file=txt_log, flush=True)

            val_batchcount+=1

        torch.save(model.state_dict(), model_name)


    # Replaced with tensorboard logging
    # x_data = list(range(epochs))
    # # Plot loss function
    # plt.scatter(x_data, train_loss_values, c='r', label='data')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('Training Loss')
    # plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Running training on the Block JNet')
    parser.add_argument('--run_name', '-r', metavar='RUNNAME', nargs='+', help='Run Name', required=True)
    parser.add_argument('--epochs', '-e', metavar='EPS', nargs='+', help='Number of epochs to train', required=True)
    parser.add_argument('--batch', '-b', metavar='BATCHSIZE', nargs='+', help='Size of batch', required=True)
    parser.add_argument('--lr', '-l', metavar='LR', nargs='+', help='Learning Rate', required=True)
    parser.add_argument('--scaling_factor', '-s', metavar='SCALEFACTOR', nargs='+', help='Scaling factor (amount of upsampling to do)', required=True)
    
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

    #Create Model
    model = jnet.TCAJNet(upsampling_layers=int(np.log2(scale_factor)))

    # Create a SummaryWriter for logging
    suffix = f"jnet_run_name_{run_name}"
    train_writer = SummaryWriter(log_dir="logs/train_logs", filename_suffix=suffix)
    val_writer = SummaryWriter(log_dir="logs/val_logs", filename_suffix=suffix)
    
    lf = open('logs/txt_logs/'+str(run_name)+'_log.txt', 'w+')

    model_name = 'models/model_' + run_name + '.pth'
    
    train_model(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            train_writer=train_writer,
            val_writer=val_writer,
            model_name=model_name,
            txt_log=lf,
            scale_factor=scale_factor)
   
    train_writer.flush()
    val_writer.flush()
    
    torch.save(model.state_dict(), model_name)

    # Close the SummaryWriter
    train_writer.close()
    val_writer.close()

# To see the tensorboard logs, run the following command in the terminal:
# tensorboard --logdir=logs
