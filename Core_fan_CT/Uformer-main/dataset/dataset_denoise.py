import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        ds_factor = 64  # 64,32,16,8,4
        mode = 'train'
        saved_path = rgb_dir

        # 加载npy文件
        train_data_Name = saved_path + 'CT_' + mode + '1_CT-FBP-fan/FBP_1.npy'
        train_full_sampled_data = np.load(train_data_Name)
        for i in range(7):  # eight files train labels
            train_data_Name = saved_path + 'CT_' + mode + str(i + 2) + '_CT-FBP-fan/FBP_1.npy'
            train_full_sampled_data_tmp = np.load(train_data_Name)
            train_full_sampled_data = np.concatenate((train_full_sampled_data, train_full_sampled_data_tmp))
        print('Train data shape', np.array(train_full_sampled_data).shape)
        self.target_ = train_full_sampled_data

        # load FBP images
        train_data_Name_FBP = saved_path + 'CT_' + mode + '1_CT-FBP-fan/FBP_' + str(ds_factor) + '.npy'
        train_full_sampled_data_FBP = np.load(train_data_Name_FBP)
        for i in range(7):  # eight files train labels
            train_data_Name_FBP = saved_path + 'CT_' + mode + str(i + 2) + '_CT-FBP-fan/FBP_' + str(ds_factor) + '.npy'
            train_full_sampled_data_FBP_tmp = np.load(train_data_Name_FBP)
            train_full_sampled_data_FBP = np.concatenate((train_full_sampled_data_FBP, train_full_sampled_data_FBP_tmp))
        print('Train FBP data shape', np.array(train_full_sampled_data_FBP).shape)
        self.input_ = train_full_sampled_data_FBP
        self.img_options=img_options

        self.tar_size = len(self.target_)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        noisy, clean = torch.from_numpy(self.input_[index]), torch.from_numpy(self.target_[index])
        noisy = torch.unsqueeze(noisy, 0)
        clean = torch.unsqueeze(clean, 0)
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:,r:r + ps, c:c + ps]
        noisy = noisy[:,r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy
##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        ds_factor = 64  # 64,32,16,8,4
        mode = 'val'
        saved_path = rgb_dir

        # test data
        train_data_Name = saved_path + 'CT_' + mode + '_CT-FBP-fan/FBP_1.npy'
        train_full_sampled_data = np.load(train_data_Name)
        print('Test full_sampled data shape', np.array(train_full_sampled_data).shape)
        self.target_ = train_full_sampled_data

        test_data_Name_FBP = saved_path + 'CT_' + mode + '_CT-FBP-fan/FBP_' + str(ds_factor) + '.npy'
        test_full_sampled_data_FBP = np.load(test_data_Name_FBP)
        print('Test FBP data shape', np.array(test_full_sampled_data_FBP).shape)
        self.input_ = test_full_sampled_data_FBP
        self.tar_size = len(self.target_)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        noisy, clean = torch.from_numpy(self.input_[index]), torch.from_numpy(self.target_[index])
        noisy = torch.unsqueeze(noisy, 0)
        clean = torch.unsqueeze(clean, 0)
        return clean, noisy

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform
        ds_factor = 64  # 64,32,16,8,4
        mode = 'test'
        saved_path = rgb_dir

        # test data
        train_data_Name = saved_path + 'CT_' + mode + '_CT-FBP-fan/FBP_1.npy'
        train_full_sampled_data = np.load(train_data_Name)
        print('Test full_sampled data shape', np.array(train_full_sampled_data).shape)
        self.target_ = train_full_sampled_data

        test_data_Name_FBP = saved_path + 'CT_' + mode + '_CT-FBP-fan/FBP_' + str(ds_factor) + '.npy'
        test_full_sampled_data_FBP = np.load(test_data_Name_FBP)
        print('Test FBP data shape', np.array(test_full_sampled_data_FBP).shape)
        self.input_ = test_full_sampled_data_FBP
        self.tar_size = len(self.target_)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        noisy, clean = torch.from_numpy(self.input_[index]), torch.from_numpy(self.target_[index])
        noisy = torch.unsqueeze(noisy, 0)
        clean = torch.unsqueeze(clean, 0)
        return clean, noisy


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)