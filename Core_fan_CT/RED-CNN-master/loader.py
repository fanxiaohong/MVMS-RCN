import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient,ds_factor, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        # input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        # target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train': # CT_test_CT-FBP(-fan)
            train_data_Name = saved_path+'CT_'+mode+'1_CT-FBP/FBP_1.npy'
            train_full_sampled_data = np.load(train_data_Name)
            for i in range(7):  # eight files train labels
                train_data_Name = saved_path+'CT_'+mode+str(i+2)+'_CT-FBP/FBP_1.npy'
                train_full_sampled_data_tmp = np.load(train_data_Name)
                train_full_sampled_data = np.concatenate((train_full_sampled_data, train_full_sampled_data_tmp))
            print('Train data shape', np.array(train_full_sampled_data).shape)
            self.target_ = train_full_sampled_data

            # load FBP images
            train_data_Name_FBP = saved_path+'CT_'+mode+'1_CT-FBP/FBP_' + str(ds_factor)+'.npy'
            train_full_sampled_data_FBP = np.load(train_data_Name_FBP)
            for i in range(7):  # eight files train labels
                train_data_Name_FBP = saved_path+'CT_'+mode+str(i+2)+'_CT-FBP/FBP_' + str(ds_factor)+'.npy'
                train_full_sampled_data_FBP_tmp = np.load(train_data_Name_FBP)
                train_full_sampled_data_FBP = np.concatenate((train_full_sampled_data_FBP, train_full_sampled_data_FBP_tmp))
            print('Train FBP data shape', np.array(train_full_sampled_data_FBP).shape)
            self.input_ = train_full_sampled_data_FBP

        elif mode =='test':
            # test data
            train_data_Name = saved_path+'CT_'+mode+'_CT-FBP/FBP_1.npy'
            train_full_sampled_data = np.load(train_data_Name)
            print('Test full_sampled data shape', np.array(train_full_sampled_data).shape)
            self.target_ = train_full_sampled_data

            test_data_Name_FBP = saved_path+'CT_'+mode+'_CT-FBP/FBP_' + str(ds_factor)+'.npy'
            test_full_sampled_data_FBP = np.load(test_data_Name_FBP)
            print('Test FBP data shape', np.array(test_full_sampled_data_FBP).shape)
            self.input_ =  test_full_sampled_data_FBP

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            print('Test beginning')
            return (input_img, target_img)


def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(mode='train', load_mode=0,
               saved_path=None, test_patient='test',ds_factor=None,patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient,ds_factor, patch_n, patch_size, transform)
    if mode == 'train':
        data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif mode =='test':
        data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader
