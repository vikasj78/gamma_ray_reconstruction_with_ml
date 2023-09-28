#!/usr/bin/env python
# coding: utf-8
import os
import glob
import os.path as osp
import numpy as np
import pandas as pd
import h5py
import time
import tables as tb
from matplotlib import pyplot as plt
import itertools as it
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph

import torch
from torch_geometric.data import Dataset

class process_hess_dataset:
    def __init__(self, fin=None, cr_type=None):
        self.fin = fin
        self.cr_type = cr_type
        
    def get_hess_geom(self):
        #hess1_cam = self.fin.get_node('/configuration/instrument/telescope/camera/geometry_HESS-I')
        #hess2_cam = self.fin.get_node('/configuration/instrument/telescope/camera/geometry_HESS-II')
        hess1_cam = self.fin.get_node('/configuration/instrument/telescope/camera/geometry_0')
        hess2_cam = self.fin.get_node('/configuration/instrument/telescope/camera/geometry_1')
        hess1_cam_geom_xc = np.array(hess1_cam.col('pix_x'))
        hess1_cam_geom_yc = np.array(hess1_cam.col('pix_y'))
        hess2_cam_geom_xc = np.array(hess2_cam.col('pix_x'))
        hess2_cam_geom_yc = np.array(hess2_cam.col('pix_y'))
        tel_loc = {'ct1': np.array([-0.16, -85.04, 0.97]), #x, y, z in cm
                   'ct2': np.array([85.07, -0.37, 0.33]),
                   'ct3': np.array([0.24, 85.04, -0.82]),
                   'ct4': np.array([-85.04, 0.28, -0.48]),
                   'ct5': np.array([0., 0., 0.])}
        cam_pixels_in_array = dict()
        #calculate pixel coordinates from the center of the array - no z axis at the moment
        for tel in tel_loc.keys():
            if(tel != 'ct5'):
                cam_pixels_in_array[tel] = np.array([tel_loc[tel][0] + hess1_cam_geom_xc, tel_loc[tel][1] + hess1_cam_geom_yc])
            elif(tel == 'ct5'):
                cam_pixels_in_array[tel] = np.array([tel_loc[tel][0] + hess2_cam_geom_xc, tel_loc[tel][1] + hess2_cam_geom_yc])
            else:
                print("what kind of camera is that! please check")
        return cam_pixels_in_array

    def get_tel_nodes(self):
        tel_nodes = dict()
        for inum, node in enumerate(self.fin.get_node('/dl1/event/telescope/images')):
            tel_name = f'ct{inum+1}'
            tel_nodes[tel_name] = node
        return tel_nodes

    def get_data_list(self):
        tel_nodes = self.get_tel_nodes()
        cam_pixels_in_array = self.get_hess_geom()
        data_list = list()
        trig_events = self.fin.get_node('/dl1/event/subarray/trigger')
        count = 0
        for ev_num,ev in enumerate(trig_events):
            obs_id = ev['obs_id']
            event_id = ev['event_id']
            #print(ev_num, obs_id, event_id)
            pe = list()
            x = list()
            y = list()
            for tel_num, tel in enumerate(ev['tels_with_trigger']):
                if(tel and tel_num < 5): #only doing ct1-4 for now, just for testing..
                    tel_name = f'ct{tel_num+1}'
                    #print(ev_num, obs_id, event_id, tel_name)
                    image = tel_nodes[tel_name].read_where(f'(obs_id == {obs_id}) & (event_id == {event_id})')['image']
                    #print(image.shape)
                    if (image.shape[0] == 0): #don't know why it is stored as triggered than!
                        continue
                    elif (image.shape[0] > 1): #don't know why this happens at all as well.
                        image = image[0]
                        #print('> 1', image.shape)
                    if (image.sum() < 100):
                        continue
                    image = image.flatten()
                    if (image.shape[0] != cam_pixels_in_array[tel_name][0].shape[0]):
                        count += 1
                        print("this should not happen a bug in h5 file", tel_name, 'image_shape:', image.shape[0], cam_pixels_in_array[tel_name][0].shape)
                        continue

                    pix_pe_theshold_mask = image > 5
                    pe.append(image[pix_pe_theshold_mask])
                    x.append(cam_pixels_in_array[tel_name][0][pix_pe_theshold_mask])
                    y.append(cam_pixels_in_array[tel_name][1][pix_pe_theshold_mask])
                    #print(image[pix_pe_theshold_mask].shape, tel_name)
            if not pe:
                continue
            pe = np.log10(np.concatenate(pe).flatten())
            x = np.concatenate(x).flatten()
            y = np.concatenate(y).flatten()
            sum_pe = pe.sum()
            if (sum_pe < 10):
                continue
            xcom = (x * pe).sum()/sum_pe
            ycom = (y * pe).sum()/sum_pe
            x = x - xcom
            y = y - ycom
            # print(xcom, ycom)
            # print(x.max(), y.max(), pe.max())
            # print(x, y, pe)
            # break
            nodes = torch.t(torch.tensor(np.array((x,y,pe)), dtype=torch.float))
            if(self.cr_type == 'gamma'):
                data = Data(x=nodes, edge_index=None, y=1)
                data.edge_index  = knn_graph(
                                    data.x[:, [0,1,2]], k=5)
            else:
                data = Data(x=nodes, edge_index=None, y=0)
                data.edge_index = knn_graph(
                                    data.x[:, [0,1,2]], k=5)
          
            data_list.append(data)
        return data_list

# #### let's define the dataset class

class MyDataset(Dataset, process_hess_dataset):
    def __init__(self, root, name, indir, num_files, transform=None, pre_transform=None, pre_filter=None):
        process_hess_dataset.__init__(self)
        self.root = root
        self.name = name
        self.indir = indir
        self.num_files = num_files
        super().__init__(root, transform, pre_transform, pre_filter)
        
    
    @property
    def raw_file_names(self):
        cr_types = ['gamma','proton']
        in_file_list = list()
        for cr_type in cr_types:
            in_file_list_temp = glob.glob(self.indir + f'/{cr_type[0]}*')
            in_file_list += in_file_list_temp[0:self.num_files]
        return in_file_list

    @property
    def processed_file_names(self):
        cr_types = ['gamma','proton']
        processed_file_list = list()
        for cr_type in cr_types:
            processed_file_list += glob.glob(self.processed_dir + f'/{cr_type[0]}*')
        return processed_file_list

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    def len(self):
        return len(self.processed_file_names)

    def get(self, cr_type, idx):
        data = torch.load(osp.join(self.processed_dir, f'{cr_type}_{idx}.pt'))
        return data

    def process(self):
        #torch.save(self.collate(self.data_list), self.processed_paths[0])
        gamma_idx = 0
        proton_idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            fin = tb.open_file(raw_path, mode="r")
            #data = self.get_data_list(fin, self.get_hess_geom(fin), self.get_tel_nodes(fin), cr_type)
            print('processing..', raw_path)
            cr_type = None
            if 'gamma' in raw_path.split('/')[-1]:
                cr_type = 'gamma'
                torch.save(process_hess_dataset(fin, cr_type).get_data_list(), osp.join(self.processed_dir, f'{cr_type}_{gamma_idx}.pt'))
                #break
                gamma_idx += 1
            elif 'proton' in raw_path.split('/')[-1]:
                cr_type = 'proton'
                torch.save(process_hess_dataset(fin, cr_type).get_data_list(), osp.join(self.processed_dir, f'{cr_type}_{proton_idx}.pt'))
                proton_idx += 1
            fin.close()
                
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

def get_event_images(dataset, cr_type, file_num, event_num):
    file_number = 0
    event_number = 0
    for raw_path in dataset.raw_paths:
        if cr_type in raw_path.split('/')[-1]:
            if(file_number == file_num):
                fin = tb.open_file(raw_path, mode="r")
                gamma_dataset = process_hess_dataset(fin, cr_type)
                cam_pixels_in_array = gamma_dataset.get_hess_geom() 
                tel_nodes = gamma_dataset.get_tel_nodes()
                trig_events = fin.get_node('/dl1/event/subarray/trigger')
                defualt_value_pe = np.zeros((1,960))

                for ev_num,ev in enumerate(trig_events):
                    obs_id = ev['obs_id']
                    event_id = ev['event_id']
                    pe = list()
                    tel_images = dict()
                    for tel_num, tel in enumerate(ev['tels_with_trigger']):
                        if(tel and tel_num < 4): #only doing ct1-4 for now, just for testing..
                            tel_name = f'ct{tel_num+1}'
                            image = tel_nodes[tel_name].read_where(f'(obs_id == {obs_id}) & (event_id == {event_id})')['image']
                            tel_images[tel_name] = image
                            if (image.shape[0] == 0): #don't know why it is stored as triggered than!
                                continue
                            elif (image.shape[0] > 1): #don't know why this happens at all as well.
                                image = image[0]
                            if (image.sum() < 100):
                                continue
                            image = image.flatten()
                            if (image.shape[0] != cam_pixels_in_array[tel_name][0].shape[0]):
                                count += 1
                                print("this should not happen a bug in h5 file", tel_name, 'image_shape:', image.shape[0], cam_pixels_in_array[tel_name][0].shape)
                                continue

                            pix_pe_theshold_mask = image > 5
                            pe.append(image[pix_pe_theshold_mask])
                    if not pe:
                        continue
                    pe = np.concatenate(pe).flatten()
                    if (pe.sum() < 1000):
                        continue
                    if(event_number == event_num):
                        for tel in range(1,5):
                            tel_key = f'ct{tel}'
                            if tel_key not in tel_images.keys():
                                tel_images[tel_key] = defualt_value_pe
                        print( file_number, event_number)
                        return cam_pixels_in_array, tel_images
                    else:
                        event_number += 1 
                fin.close()
            else:
                file_number += 1
                
def main():

    #indir = '/home/saturn/caph/mpp228/HESS_data/HESS_data_MC/sim_telarray/phase2d/NSB1.00/Desert/Proton_Electron_Gamma-diffuse/20deg/180deg/0.0deg-ws0/Data_h5'
    #outdir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/test_dataset'
    indir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/phase2d2_bbruno/'
    outdir = '/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/phase2d2_dataset'
    dataset_name = 'test'

    dataset = MyDataset(outdir,dataset_name,indir,1)


if __name__ == "__main__":
    main()





