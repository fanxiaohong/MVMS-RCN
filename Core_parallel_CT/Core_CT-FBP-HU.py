import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
import torchvision
from data_loader_HU import *
from solver_CT_FBP_HU import Solver_CT_FBP
###########################################################################################
# parameter
parser = ArgumentParser(description='CT-FBP')
parser.add_argument('--net_name', type=str, default='CT-FBP-HU', help='name of net')
parser.add_argument('--model_dir', type=str, default='model_CT', help='model_MRI,model_CT,trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log_CT', help='log_MRI,log_CT')
parser.add_argument('--batch_size', type=int, default=1, help='MRI=1,CT=1')
parser.add_argument('--ds_factor', type=int, default=12, help='from {48,24,12, 8, 6, 4}')
parser.add_argument('--CT_test_name', type=str, default='CT_test', help='name of CT test set')
parser.add_argument('--num_work', type=int, default=4, help='4,1')
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--run_mode', type=str, default='test', help='train、test')
parser.add_argument('--save_image_mode', type=int, default=0, help='save 1, 2 npy file,not 0 in test')
args = parser.parse_args()
#########################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###########################################################################################
# data loading
train_loader,val_loader,test_loader = CT_dataloader(args.num_work,args.batch_size,args.run_mode)
#######################################################################################
solver = Solver_CT_FBP(test_loader, args, device)
solver.test()
#########################################################################################
