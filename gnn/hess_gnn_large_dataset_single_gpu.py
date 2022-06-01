#!/usr/bin/env python
# coding: utf-8
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from prepare_hess_dataset_large_hdf5 import MyDataset

import torch
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn import Linear 
from torch.nn import BatchNorm1d, Linear, Dropout

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import TAGConv, global_add_pool, global_max_pool

from pytorch_lightning.core.lightning import LightningModule
from GNN import GNN

torch.cuda.empty_cache()

#define the model
class GCN(GNN):
    def __init__(self, nb_inputs, nb_outputs, nb_intermediate=128, dropout_ratio=0.3):
        """ConvNet model.
        Args:
            nb_inputs (int): Number of input features, i.e. dimension of input
                layer.
            nb_outputs (int): Number of prediction labels, i.e. dimension of
                output layer.
            nb_intermediate (int): Number of nodes in intermediate layer(s)
            dropout_ratio (float): Fraction of nodes to drop
        """
        # Base class constructor
        super().__init__(nb_inputs, nb_outputs)

        # Member variables
        self.nb_intermediate = nb_intermediate
        self.nb_intermediate2 = 6 * self.nb_intermediate

        # Architecture configuration
        self.conv1 = TAGConv(self.nb_inputs, self.nb_intermediate, 2)
        self.conv2 = TAGConv(self.nb_intermediate, self.nb_intermediate, 2)
        self.conv3 = TAGConv(self.nb_intermediate, self.nb_intermediate, 2)

        self.batchnorm1 = BatchNorm1d(self.nb_intermediate2)

        self.linear1 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear2 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear3 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear4 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear5 = Linear(self.nb_intermediate2, self.nb_intermediate2)

        self.drop1 = Dropout(dropout_ratio)
        self.drop2 = Dropout(dropout_ratio)
        self.drop3 = Dropout(dropout_ratio)
        self.drop4 = Dropout(dropout_ratio)
        self.drop5 = Dropout(dropout_ratio)

        self.out = Linear(self.nb_intermediate2, self.nb_outputs)

    def forward(self, data: Data) -> Tensor:
    #def forward(self, x, edge_index, batch):
        """Model forward pass.
        Args:
            data (Data): Graph of input features.
        Returns:
            Tensor: Model output.
        """

        #Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph convolutional operations
        x = F.leaky_relu(self.conv1(x, edge_index))
        x1 = torch.cat([
            global_add_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=1)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x2 = torch.cat([
            global_add_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=1)

        x = F.leaky_relu(self.conv3(x, edge_index))
        x3 = torch.cat([
            global_add_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=1)

        # Skip-cat
        x = torch.cat([x1, x2, x3], dim=1)

        # Batch-normalising intermediate features
        x = self.batchnorm1(x)
        # Post-processing
        x = F.leaky_relu(self.linear1(x))
        x = self.drop1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.drop2(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.drop3(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.drop4(x)
        x = F.leaky_relu(self.linear5(x))
        x = self.drop5(x)
        # Read-out
        x = self.out(x)
        #x = F.softmax(x, dim=1)
        return x

    @torch.no_grad()
    def inference(self, data: Data) -> Tensor:
        return self.forward(data)
        
def train(model, criterion, optimizer, file_ini, train_test_split, batch_size, device):
    model.train()
    #load the test and train datasets
    cr_types = ['gamma','proton']
    for idx in range(file_ini,train_test_split):
        train_dataset = list()
        for cr_type in cr_types:
            #print(cr_type, idx)
            train_dataset += dataset.get(cr_type,idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y[:,-1].unsqueeze(1))
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
    #         loss_temp += loss.item()
    #     total_samples += len(train_loader.dataset)
    # train_loss = loss_temp/total_samples #the final loss at each epoch
    # return train_loss

def model_eval(model, criterion, loader, device):
    model.eval()
    correct = 0
    loss_temp = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device) 
        out = model(data)
        loss = criterion(out, data.y[:,-1].unsqueeze(1))
        loss_temp += loss.item()
        pred = torch.round(torch.sigmoid(out))
        correct += int((pred == data.y[:,-1].unsqueeze(1)).sum())
    return loss_temp, correct, len(loader.dataset)  

def get_performance(model, criterion, ini_file, end_file, batch_size, device):
    cr_types = ['gamma','proton']
    correct = 0 
    total_samples = 0
    final_loss = 0
    for idx in range(ini_file,end_file):
        temp_dataset = list()
        for cr_type in cr_types:
            temp_dataset += dataset.get(cr_type,idx)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
        loss_temp, correct_temp, total_samples_temp  = model_eval(model, criterion, temp_loader, device)
        final_loss += loss_temp
        correct += correct_temp
        total_samples += total_samples_temp
    final_loss = final_loss/total_samples
    acc = correct/total_samples
    return final_loss, acc

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min


if __name__ == '__main__':

    # #### get the dataset and have a look
    analysis_type = 'mono'
    indir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/'
    outdir = '/home/saturn/caph/mppi067h/graph_datasets/phase2d3/' +  analysis_type
    out_model_dir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/trained_models/phase2d3/' + analysis_type
    dataset_name = 'test'
    best_model_path = out_model_dir + f'/model_best.pth'
    
    dataset = MyDataset(outdir,dataset_name,indir,1)
    start_epoch = 1
    num_epochs = 200
    batch_sizes = {'mono': 1024,
                   'stereo': 512,
                    'hybrid': 256}#for mono = 1024, for stereo 512, for hybrid 256
    batch_size = batch_sizes[analysis_type]
    file_ini = 0 
    file_end = 50
    learning_rate = 1e-3
    #run the model
    train_valid_split = int((file_end - file_ini)*0.8)
    print('train_validation_split', file_ini, train_valid_split, file_end)
    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(nb_inputs=3, nb_outputs=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    valid_loss_min = np.Inf
    if(os.path.exists(best_model_path)):
        model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)  
        start_epoch = start_epoch + 1
    for epoch in range(start_epoch, num_epochs):
        print('starting epoch', epoch)
        train(model, criterion, optimizer, file_ini, train_valid_split, batch_size, device)
        train_loss, train_acc = get_performance(model, criterion, file_ini, train_valid_split, batch_size, device)
        valid_loss, valid_acc = get_performance(model, criterion, train_valid_split, file_end, batch_size, device)
        
        print(f'Epoch: {epoch:03d}, train loss: {train_loss*1e5:.4f}, validation loss: {valid_loss*1e5:.4f}, Train Acc: {train_acc:.4f}, Validation Acc: {valid_acc:.4f}')
        checkpoint_path = out_model_dir + f'/model_{epoch}_loss_tr{train_loss*1e5:.2f}_loss_val{valid_loss*1e5:.2f}_tr{train_acc:.2f}_val{valid_acc:.2f}.pth'
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
