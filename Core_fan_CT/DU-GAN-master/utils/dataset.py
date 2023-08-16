import torch
import torch.utils.data as tordata
import os.path as osp
import numpy as np
from functools import partial


class CTPatchDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range, transforms=None):
        self.transforms = transforms
        hu_min, hu_max = hu_range
        data = torch.from_numpy(np.load(npy_root).astype(np.float32) - 1024)
        # normalize to [0, 1]
        data = (torch.clamp(data, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
        # data= data/(hu_max - hu_min)
        self.low_doses, self.full_doses = data[0], data[1]

    def __getitem__(self, index):
        low_dose, full_dose = self.low_doses[index], self.full_doses[index]
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.low_doses)


data_root = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset')
dataset_dict = {
    'cmayo_train_64': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/train_64.npy')),
    'cmayo_test_512': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/test_512.npy')),

    'train_ds24': partial(CTPatchDataset, npy_root='../data/CT/parallel/train_ds24.npy'),
    'test_ds24': partial(CTPatchDataset, npy_root='../data/CT/parallel/test_ds24.npy'),

    'train_ds12': partial(CTPatchDataset, npy_root='../data/CT/parallel/train_ds12.npy'),
    'test_ds12': partial(CTPatchDataset, npy_root='../data/CT/parallel/test_ds12.npy'),

    'train_ds8': partial(CTPatchDataset, npy_root='../data/CT/parallel/train_ds8.npy'),
    'test_ds8': partial(CTPatchDataset, npy_root='../data/CT/parallel/test_ds8.npy'),

    'train_ds6': partial(CTPatchDataset, npy_root='../data/CT/parallel/train_ds6.npy'),
    'test_ds6': partial(CTPatchDataset, npy_root='../data/CT/parallel/test_ds6.npy'),

    'train_ds4': partial(CTPatchDataset, npy_root='../data/CT/parallel/train_ds4.npy'),
    'test_ds4': partial(CTPatchDataset, npy_root='../data/CT/parallel/test_ds4.npy'),

    'train_ds64_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/train_ds64.npy'),
    'test_ds64_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/test_ds64.npy'),

    'train_ds32_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/train_ds32.npy'),
    'test_ds32_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/test_ds32.npy'),

    'train_ds16_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/train_ds16.npy'),
    'test_ds16_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/test_ds16.npy'),

    'train_ds8_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/train_ds8.npy'),
    'test_ds8_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/test_ds8.npy'),

    'train_ds4_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/train_ds4.npy'),
    'test_ds4_fan': partial(CTPatchDataset, npy_root='../data/CT/fan/test_ds4.npy'),

}
