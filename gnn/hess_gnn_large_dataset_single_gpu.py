#!/usr/bin/env python
# coding: utf-8
import os
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
            #print(out.shape, out.dtype, data.y.unsqueeze(1).shape, data.y.unsqueeze(1).dtype)
            loss = criterion(out, data.y.unsqueeze(1).float())  # Compute the loss.
            #loss = criterion(torch.squeeze(out), data.y)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
    return loss #the final loss at each epoch


def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device) 
        #out = model.module.inference(data)
        out = model(data)
        #pred = out.argmax(dim=1)  # Use the class with highest probability.
        #correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        pred = torch.round(torch.sigmoid(out))
        correct += int((pred == data.y.unsqueeze(1)).sum())
        
    return correct, len(loader.dataset)

def get_accuracy(model, ini_file, end_file, batch_size, device):
    cr_types = ['gamma','proton']
    correct = 0 
    total_samples = 0
    for idx in range(ini_file,end_file):
        temp_dataset = list()
        for cr_type in cr_types:
            temp_dataset += dataset.get(cr_type,idx)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
        correct_temp, total_samples_temp  = test(model, temp_loader, device)
        correct += correct_temp
        total_samples += total_samples_temp
    acc = correct/total_samples
    return acc


if __name__ == '__main__':

    # #### get the dataset and have a look
    indir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/phase2d2_bbruno_large/'
    outdir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/phase2d2_dataset_large'
    out_model_dir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/trained_models_single_gpu2/'
    dataset_name = 'test'
    
    dataset = MyDataset(outdir,dataset_name,indir,1)
    num_epochs = 30
    batch_size = 256
    file_ini = 0 
    file_end = 30
    #run the model
    train_test_split = int((file_end - file_ini)*0.8)
    print('train_test_split', file_ini, train_test_split, file_end)
    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(nb_inputs=3, nb_outputs=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, num_epochs):
        print('starting epoch', epoch)
        loss = train(model, criterion, optimizer, file_ini, train_test_split, batch_size, device)
        train_acc = get_accuracy(model, file_ini, train_test_split, batch_size, device)
        test_acc = get_accuracy(model, train_test_split, file_end, batch_size, device)
        
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        torch.save(model.state_dict(),out_model_dir + f'/model_{epoch}_loss{loss:.2f}_tr{train_acc:.2f}_te{test_acc:.2f}.pth')
