import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
from skimage.measure import compare_ssim as ssim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import SimpleITK as sitk
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
import scipy.io as sio
from dataset.dataset_denoise import *
import utils
import math
from model import UNet,Uformer
from time import time
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import cv2

parser = argparse.ArgumentParser(description='Image denoising evaluation on CT')
parser.add_argument('--input_dir', default='../../result/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/denoising/CT/Uformer_B-fan-ds32/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='../train/logs/denoising/CT/Uformer_B_/models-fan-ds32/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=1, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
len_testset = test_loader.__len__()
print("sizeof test set: ", len_testset)

model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

##########################################################################
def compute_measure(y_gt, y_pred, data_range):
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_rmse)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))
##########################################################################
def psnr(img1, img2):
    # img1.astype(np.float32)
    # img2.astype(np.float32)
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
##########################################################################
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
with torch.no_grad():
    psnr_dataset = []
    psnr_model_init = []
    SSIM_total = []
    RMSE_total = []
    PSNR_total = []
    image_num = 0
    time_all = 0
    x_test_all = []
    for ii, data_test in enumerate((test_loader), 0):
        image_num = image_num + 1
        target = data_test[0].cuda()
        input_ = data_test[1].cuda()
        start = time()
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
        end = time()
        time_all = time_all + end - start
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())

        restored = torch.squeeze(restored)
        restored = restored.cpu().data.numpy()#.reshape(target.shape[2], target.shape[3],3)

        target = target.cpu().data.numpy().reshape(target.shape[2], target.shape[3])

        restored = np.mean(restored, axis=0)

        rec_RMSE = compute_measure(target, restored, 1)
        rec_PSNR = psnr(restored * 255.0, target * 255.0)
        rec_SSIM = ssim(restored, target, data_range=1)
        SSIM_total.append(rec_SSIM)
        RMSE_total.append(rec_RMSE)
        PSNR_total.append(rec_PSNR)
        x_test_all.append(restored)  # assemble

    x_test_all = np.array(x_test_all)
    out = sitk.GetImageFromArray(x_test_all * 4096 - 1024)
    save_npy_name = args.result_dir + 'Output.nii.gz'
    sitk.WriteImage(out, save_npy_name)


    psnr_dataset = sum(psnr_dataset) / len_testset
    psnr_model_init = sum(psnr_model_init) / len_testset
    print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))
    PSNR_mean = np.array(PSNR_total).mean()
    PSNR_std = np.array(PSNR_total).std()
    SSIM_mean = np.array(SSIM_total).mean()
    SSIM_std = np.array(SSIM_total).std()
    RMSE_mean = np.array(RMSE_total).mean()
    RMSE_std = np.array(RMSE_total).std()
    print("Mean run time for test is %.6f, PSNR is %.3f-%.3f, "
          "SSIM is %.5f-%.5f, RMSE is %.5f-%.5f" % (time_all / image_num,
              PSNR_mean, PSNR_std, SSIM_mean, SSIM_std, RMSE_mean, RMSE_std))

    # save result
    output_file_name1 = args.result_dir + 'result_rmse_all.txt'
    output_data_rmse = [RMSE_total]
    output_file1 = open(output_file_name1, 'a')
    for fp in output_data_rmse:  # write data in txt
        output_file1.write(str(fp))
        output_file1.write(',')
    output_file1.write('\n')  # line feed
    output_file1.close()