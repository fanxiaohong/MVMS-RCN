import os
import os.path as osp
import numpy as np
import tqdm
import pandas as pd
from pydicom import dcmread
import matplotlib.pyplot as plt

ds_factor = 12  # 24,12,8,6,4


# C021, C120 ...
for phase in ['test']:
    print('generate {} data...'.format(phase))

    # test data
    train_data_Name = './CT_' + phase + '_CT-FBP/FBP_1.npy'
    train_full_sampled_data = np.load(train_data_Name)
    print('Test full_sampled data shape', np.array(train_full_sampled_data).shape)
    target_ = train_full_sampled_data

    test_data_Name_FBP = './CT_' + phase + '_CT-FBP/FBP_' + str(ds_factor) + '.npy'
    test_full_sampled_data_FBP = np.load(test_data_Name_FBP)
    print('Test FBP data shape', np.array(test_full_sampled_data_FBP).shape)
    input_ = test_full_sampled_data_FBP

    # for phase in ['test']:
    patches = []
    for j in tqdm.trange(len(input_)):
    # for j in tqdm.trange(1):
        f_ps = target_[j,:,:]*4096
        l_ps = input_[j,:,:]*4096

        # plt.imshow(l_ps)
        # plt.show()

        patches.append(np.array([l_ps, f_ps]))
    patches = np.array(patches).reshape((-1, 2, 1, f_ps.shape[1], f_ps.shape[1])).transpose((1, 0, 2, 3, 4))

    print(patches.shape)
    # plt.imshow(patches[0,:,:,:,:].reshape(512,512))
    # plt.show()

    print('process {} patches...'.format(phase))
    np.random.seed(0)
    print('save {} patches...'.format(phase))
    np.save(osp.join('./', '{}'.format(phase)), patches[:, :, :, :, :])
    print('complete save {} patches...'.format(phase))

