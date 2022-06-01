import os, sys
import uproot
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data

import torch
from torch_geometric.data import InMemoryDataset


class MyDataset(InMemoryDataset):
    def __init__(self, root, name, data_list=None, transform=None):
        self.data_list = data_list
        self.name = name
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

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

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])
        
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

def main():
    # #### get camera coordinate file
    hess_camera = pd.read_csv('hess_camera.dat', delim_whitespace=True, comment="#", error_bad_lines=False)
    hess_camera.iloc[[0, -1]]
    camera_coord = hess_camera[[ '0', '0.1']].copy()
    camera_coord.reset_index(drop=True, inplace=True)
    camera_coord = camera_coord.rename(columns={'0': 'x', '0.1':'y'})
    print(camera_coord.shape)
    camera_coord.head()

    # #### get the data files
    event_types = {'proton':['proton.root'],
                    'gamma':['gamma.root'],
                    }
    varibales = ['EventFolder','FlattenedDataT1','FlattenedDataT2','FlattenedDataT3','FlattenedDataT4','MCTrueEnergy','HillasImageAmplitude']
    selection = 'EventFolder == 1'
    for key in event_types.keys():
        if(os.path.isfile(f'{key}.pkl')):
            event_types[key].append(pd.read_pickle(f'{key}.pkl'))
        else:
            fin = uproot.open(event_types[key][0])
            tree = fin['ParTree_Preselect_Postselect']
            df = tree.arrays(varibales,library="pd")
            df = df.query(selection)
            event_types[key].append(df)
            df.to_pickle(f"{key}.pkl")


    # #### create camera-wise arrays with shower image data
    num_pixels = 960 #CT 1-4
    num_tels = 4
    for key in event_types.keys():
        data_final = np.array([event_types[key][1].iloc[:,num_pixels*ct:num_pixels*(ct+1)].to_numpy() for ct in range(num_tels)])
        data_final = np.einsum('kij->ikj', data_final)
        event_types[key].append(data_final)
        print(data_final.shape)


    # #### plotting an example event
    example_event = event_types['gamma'][2][1]
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    for i,ax in enumerate(axs.flatten()):
        ax_temp = ax.scatter(camera_coord['x'], camera_coord['y'], c=example_event[i])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(ax_temp, cax=cbar_ax)
    plt.show()

    for key in event_types.keys():
        ev, num_tel, num_pix = event_types[key][2].shape
        print(key)
        print('total number of data points: ', ev*num_tel*num_pix)
        print('data points with non-zero values: ', np.count_nonzero(event_types[key][2]))
        print('{0:.1f}%'.format(np.count_nonzero(event_types[key][2])*100/(ev*num_tel*num_pix)))


    # ### let's make the graph for the pytorch-geometric
    xc = camera_coord['x'].to_numpy()
    yc = camera_coord['y'].to_numpy()
    tel_loc = {'ct1': np.array([-0.16, -85.04, 0.97])*100, #x, y, z in cm
               'ct2': np.array([85.07, -0.37, 0.33])*100,
               'ct3': np.array([0.24, 85.04, -0.82])*100,
               'ct4': np.array([-85.04, 0.28, -0.48])*100}
    cam_pixels_in_array = dict()

    #calculate pixel coordinates from the center of the array - no z axis at the moment
    for tel in tel_loc.keys():
        cam_pixels_in_array[tel] = np.array([tel_loc[tel][0] + xc, tel_loc[tel][1] + yc])
    cam_pixels_in_array['ct1'].shape

    data_list = list()
    for key in event_types.keys():
        data_final = event_types[key][2]
        for ev in range(len(data_final)):
            pe = list()
            x = list()
            y = list()
            for ct_num,ct in enumerate(cam_pixels_in_array.keys()):
                pe.append(data_final[ev][ct_num][data_final[ev][ct_num] > 0])
                x.append(cam_pixels_in_array[ct][0][data_final[ev][ct_num] > 0])
                y.append(cam_pixels_in_array[ct][1][data_final[ev][ct_num] > 0])

            pe = np.concatenate(pe, axis=0).flatten()
            x = np.concatenate(x, axis=0).flatten()
            y = np.concatenate(y, axis=0).flatten()
            if(np.sum(pe) > 5000):
                max_pe = np.max(pe)
                max_pe_index = np.argmax(pe)
                #let's define the connections (edges) between the nodes here
                edges = []
                #Two pixels in any of the cameras have the differece from the highest measured signal of less than < pe_level
                #The idea is these signals are coming from the same part of the shower
                for sig in range(len(pe)):
                    ratio_pe = pe[sig]/max_pe
                    if (ratio_pe > 0.9 and ratio_pe < 1):
                        edges.append([max_pe_index,sig])
                possible_edge_comb = list(it.combinations(np.unique(np.arange(len(pe))),2))

                for i, j in possible_edge_comb:
                    dist = np.sqrt((x[j]-x[i])*(x[j]-x[i]) + (y[j]-y[i])*(y[j]-y[i]))
                    #connection is defined if:
                    #The two pixels which has seen light are < 10 cm from each other
                    #10 cm is random for now..
                    if (dist > 1.e-2 and dist < 10):
                        edges.append([i,j])
                edge_index = torch.tensor(np.array(edges), dtype=torch.long)
                #print(edge_index.shape)
                nodes = torch.t(torch.tensor(np.array((x,y,pe)), dtype=torch.float))
                if(np.array(edges).max() > len(pe)):
                    print('smothing is not right', np.array(edges).max(), len(pe))
                    print(np.array(edges), pe)
                    break
                if(key == 'gamma'):
                    data = Data(x=nodes, edge_index=edge_index.t().contiguous(), y=1)
                else:
                    data = Data(x=nodes, edge_index=edge_index.t().contiguous(), y=0)
                data_list.append(data)

    dataset = MyDataset('/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/test_data_10cm_5000pe','test',data_list)
    dataset.process()

if __name__ == "__main__":
    main()

