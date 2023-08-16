import os
import os.path as osp
import numpy as np
import tqdm
import pandas as pd
from pydicom import dcmread

# num_samples = {
#     'train': 256000,
#     'test': 64000,
# }
threshold = 0.85
stride = 32
patch_size = 64
ds_factor = 12  # 24,12,8,6,4
# stride = 32
# patch_size = 128
num_samples = {
    'train': 100000,
}
# num_samples = {
#     'train': 100000,
#     'test': 64000,
# }

# stride = 64
# patch_size = 256


def crop(ds):
    patches = []
    for left in range(0, ds.shape[0] - patch_size + 1, stride):
        for top in range(0, ds.shape[1] - patch_size + 1, stride):
            patches.append(ds[left: left + patch_size, top: top + patch_size])
    return patches


# C021, C120 ...
for phase in ['train']:
    print('generate {} data...'.format(phase))

    train_data_Name = './' + 'CT_' + phase + '1_CT-FBP/FBP_1.npy'
    train_full_sampled_data = np.load(train_data_Name)
    for i in range(7):  # eight files train labels
        train_data_Name = './' + 'CT_' + phase + str(i + 2) + '_CT-FBP/FBP_1.npy'
        train_full_sampled_data_tmp = np.load(train_data_Name)
        train_full_sampled_data = np.concatenate((train_full_sampled_data, train_full_sampled_data_tmp))
    print('Train data shape', np.array(train_full_sampled_data).shape)
    target_ = train_full_sampled_data

    # load FBP images
    train_data_Name_FBP = './' + 'CT_' + phase + '1_CT-FBP/FBP_' + str(ds_factor) + '.npy'
    train_full_sampled_data_FBP = np.load(train_data_Name_FBP)
    for i in range(7):  # eight files train labels
        train_data_Name_FBP = './' + 'CT_' + phase + str(i + 2) + '_CT-FBP/FBP_' + str(ds_factor) + '.npy'
        train_full_sampled_data_FBP_tmp = np.load(train_data_Name_FBP)
        train_full_sampled_data_FBP = np.concatenate((train_full_sampled_data_FBP, train_full_sampled_data_FBP_tmp))
    print('Train FBP data shape', np.array(train_full_sampled_data_FBP).shape)
    input_ = train_full_sampled_data_FBP

    # for phase in ['test']:
    patches_all = []
    patches = []
    for j in tqdm.trange(len(input_)):
        f_ps = crop(target_[j,:,:])
        l_ps = crop(input_[j,:,:])
        for k in range(len(f_ps)):
        #     black_percent = np.mean(np.clip(f_ps[k] - 1024, -500, 500) == -500)
        #     if black_percent < threshold:
            patches.append(np.array([l_ps[k], f_ps[k]]))
    patches = np.array(patches).reshape((-1, 2, 1, patch_size, patch_size)).transpose((1, 0, 2, 3, 4))
    # patches = np.array(patches)
    print(np.array(patches).shape)
    print('process {} patches...'.format(phase))
    for k in tqdm.trange(patches.shape[1]):  # 删掉不符合要求的patech
        black_percent = np.mean(np.clip(patches[:, k, :, :, :]*4096 - 1024, -500, 500) == -500)
        if black_percent < threshold:
            patches_all.append(patches[:, k, :, :, :])

    patches_all = np.array(patches_all).transpose((1, 0, 2, 3, 4))
    print(patches_all.shape)
    # # 保存index
    # index_file_path = osp.join('./' + phase + '_index.npy')
    # print(index_file_path)
    # if osp.exists(index_file_path):
    #     index = np.load(index_file_path)
    # else:
    #     index = np.random.choice(patches_all.shape[1], num_samples[phase], replace=False)
    #     np.save(index_file_path, index)
    index = np.random.choice(patches_all.shape[1], num_samples[phase], replace=False)
    print('save {} patches...'.format(phase))
    pathes_save = patches_all[:, index, :, :, :]*4096
    pathes_save = np.array(pathes_save)#.reshape((-1, 2, 1, patch_size, patch_size)).transpose((1, 0, 2, 3, 4))
    print(np.array(pathes_save).shape)
    np.save(osp.join('./', '{}'.format(phase)), pathes_save)
    print('complete save {} patches...'.format(phase))

