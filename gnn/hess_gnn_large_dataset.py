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
        x = F.softmax(x, dim=1)
        return x

    @torch.no_grad()
    def inference(self, data: Data) -> Tensor:
        return self.forward(data)
        
def train():
    model.train()
    #load the test and train datasets
    cr_types = ['gamma','proton']
    for idx in range(0,25):
        train_dataset = list()
        for cr_type in cr_types:
            #print(cr_type, idx)
            train_dataset += dataset.get(cr_type,idx)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            for data in train_loader:  # Iterate in batches over the training dataset.
                data = data.to(device)
                #out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
                out = model(data)
                # loss = criterion(out, data.y)  # Compute the loss.
                # loss.backward()  # Derive gradients.
                # optimizer.step()  # Update parameters based on gradients.
                # optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        #out = model(data.x, data.edge_index, data.batch)  
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct, len(loader.dataset)
    #return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def get_accuracy(ini_file, end_file):
    cr_types = ['gamma','proton']
    correct = 0 
    total_samples = 0
    for idx in range(ini_file,end_file):
        temp_dataset = list()
        for cr_type in cr_types:
            temp_dataset += dataset.get(cr_type,idx)
            temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=True)
            correct_temp, total_samples_temp  = test(temp_loader)
            correct += correct_temp
            total_samples += total_samples_temp
    acc = correct/total_samples
    return acc

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def run(rank, world_size, dataset, model_dir, num_epochs, batch_size, file_ini, file_end):
    train_test_split = int((file_end - file_ini)*0.8)
    print('train_test_split', file_ini, train_test_split, file_end)
    #print('I am in run')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    #print('rank.......................', rank)
    torch.manual_seed(12345)
    model = GCN(nb_inputs=3, nb_outputs=2).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs+1):
        #print("starting", epoch)
        model.train()
        #load the test and train datasets
        cr_types = ['gamma','proton']
        for idx in range(file_ini,train_test_split):
            train_dataset = list()
            for cr_type in cr_types:
                dataset_temp = dataset.get(cr_type,idx)
                train_dataset_temp = list(chunks(dataset_temp, int(len(dataset_temp)/world_size)))[rank]
                train_dataset += train_dataset_temp
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for bidx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
                data = data.to(rank)
                out = model(data)
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
        
        dist.barrier()          
        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        if rank == 0:# and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            correct = 0 
            total_samples = 0
            cr_types = ['gamma','proton']
            for idx in range(train_test_split,file_end):
                test_dataset = list()
                for cr_type in cr_types:
                    test_dataset += dataset.get(cr_type,idx)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                with torch.no_grad():
                    for bidx, data in enumerate(test_loader):  # Iterate in batches over the training/test dataset.
                        data = data.to(rank)
                        out = model.module.inference(data)
                        pred = out.argmax(dim=1)  # Use the class with highest probability.
                        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
                    total_samples += len(test_loader.dataset)
            test_acc = correct/total_samples
            print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
            torch.save(model.state_dict(),model_dir + f'/model_{epoch}_{test_acc:.2f}.pth')
            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss,
            # 'test_acc': test_acc,
            # }, model_dir + f'/model_{epoch}_{test_acc:.2f}.pth')
        dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':

    # #### get the dataset and have a look
    indir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/phase2d2_bbruno_large/'
    outdir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/phase2d2_dataset_large'
    out_model_dir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/trained_models_multigpu/'
    dataset_name = 'test'
    
    dataset = MyDataset(outdir,dataset_name,indir,1)
    num_epochs = 50
    batch_size = 64
    file_ini = 0 
    file_end = 30
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset, out_model_dir, num_epochs, batch_size, file_ini, file_end), nprocs=world_size, join=True)