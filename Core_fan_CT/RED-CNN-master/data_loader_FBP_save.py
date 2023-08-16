import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import scipy.io as sio
import h5py
####################################################################################
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
####################################################################################
# loader train,val,test
def dataset_loder(state,full_sampled,batch_size,num_work):
    full_sampled = torch.from_numpy((np.array(full_sampled).astype(np.float32)))
    dataset = TensorDataset(full_sampled)
    if state =='train':
        dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True,num_workers=num_work)
    else:
        dataloader = DataLoader(dataset, batch_size=1, num_workers=num_work)
    return dataloader
#####################################################################################
# ct train dataloder
def CT_dataloader(num_work,batch_size):
    # test data
    test_data_Name = './data/CT/HU/val/full_sampled.mat'
    test_full_sampled_data = sio.loadmat(test_data_Name)
    test_full_sampled_matrix = test_full_sampled_data['image_all']
    print('Test full_sampled data shape', np.array(test_full_sampled_matrix).shape)
    test_loader = DataLoader(dataset=RandomDataset(test_full_sampled_matrix, test_full_sampled_matrix.shape[0]),
                            batch_size=1, num_workers=num_work)
    return test_loader,test_loader,test_loader
