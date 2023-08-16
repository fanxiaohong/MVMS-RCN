import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
import torchvision
from data_loader_HU import *
from model_AMP_Net import AMP_Net
from solver_CT_together_parallel import Solver_CT
###########################################################################################
# parameter
parser = ArgumentParser(description='CT-new')
parser.add_argument('--net_name', type=str, default='CT-AMP-Net-K-ds12', help='name of net')
parser.add_argument('--model_dir', type=str, default='model and result parallel CT HU', help='model_MRI,model_CT,trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log_CT', help='log_MRI,log_CT')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=50, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=1, help='MRI=1,CT=1')
parser.add_argument('--layer_num', type=int, default=7, help='D,11')
parser.add_argument('--num_features', type=int, default=32, help='G,32')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--ds_factor', type=int, default=12, help='{24,12, 8, 6, 4} for parallel')
parser.add_argument('--num_work', type=int, default=4, help='4,1')
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')
parser.add_argument('--result_dir', type=str, default='result', help='save result directory')
parser.add_argument('--CT_test_name', type=str, default='CT_test', help='save name of CT test set')
parser.add_argument('--together_mode', type=str, default='single', help='all,single')
parser.add_argument('--run_mode', type=str, default='test', help='train、test')
parser.add_argument('--save_image_mode', type=int, default=0, help='save 1, not 0 in test')
parser.add_argument('--loss_mode', type=str, default='L2', help='vggloss,midloss,L1,L2,Fista-net,Ista-net')
parser.add_argument('--penalty_mode', type=str, default='None', help='Fista-net,Weight,None')
args = parser.parse_args()
#########################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###########################################################################################
# data loading
train_loader,val_loader,test_loader = CT_dataloader(args.num_work,args.batch_size,args.run_mode)
model = AMP_Net(args.layer_num)
###################################################################################
# model
model = nn.DataParallel(model)
model = model.to(device)
###################################################################################
if args.print_flag:  # print networks parameter number
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
#######################################################################################
solver = Solver_CT(model, train_loader,val_loader,test_loader, args, device, args.penalty_mode)
if args.run_mode == 'train':
    solver.train()
elif args.run_mode == 'test':
    solver.test()
#########################################################################################
