import cv2
import copy
from time import time
import math
from skimage.measure import compare_ssim as ssim
import torch
import scipy.io as sio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
##########################################################################################
# 超参数
# exact_image_str = '../../model and result parallel CT HU\Ground-truth/CT-HU-720View_CT-FBP-HU_ds_1/'  # ground truth image 所在位置
ds_factor = 32   # 64-7,32-3.2，16-1.3,8-0.4，4-0.1
TV_lambda = 3.2
# trunc_mode ='HU' # 'HU、liver, lung
# trunc_mode ='liver' # 'HU、liver, lung
trunc_mode ='HU' # 'HU、liver, lung
save_mode = 1  # 1 save , 0 don't save

##########################################################################################
exact_image_str = './result_HU/label_mat/'  # ground truth image 所在位置
predict_image_str = './result_HU/ds'+str(ds_factor)+'/100-'+str(TV_lambda)+'_mat/' # predict image image 所在位置，
# reconstruction Gauss-AMP1-5，reconstruction fast-BM3D-AMP1-5，reconstruction BM3D-AMP1-5
predict_save_name = 'im_rec'  # mat文件中的统一命名，D-AMP=x_hat1,
predict_image_str_save = './result_HU/ds'+str(ds_factor)+'/reconstruction_png_'+trunc_mode +'/'
##########################################################################################
if save_mode ==1:
    if not os.path.exists(predict_image_str_save):
        os.makedirs(predict_image_str_save)
##########################################################################################
# test image index
def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)
#########################################################################################
# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)
#####################################################################################
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
###################################################################################
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
##################################################################################
def denormalize_(image, norm_range_max, norm_range_min):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image

def trunc(mat,trunc_max,trunc_min):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    mat_scale = (mat - trunc_min)/(trunc_max - trunc_min) # scale to 0-1
    return mat_scale
##################################################################################
# test implement image, save image
print('\n')
print("CS Reconstruction Start")
filepaths = glob.glob(os.path.join(exact_image_str) + '/*.mat')
filepaths_predict = glob.glob(os.path.join(predict_image_str)+ '/*.mat*')
ImgNum = len(filepaths_predict)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
RMSE_total = []
x_test_all = []
for img_no in range(ImgNum):
    imgName = exact_image_str + str(img_no+1) + '.mat'
    # imgName = filepaths[img_no]
    # print('imgName',imgName)
    # imgName_predict = filepaths_predict[img_no]
    imgName_predict = predict_image_str + str(img_no+1)+'.mat'
    # print('imgName_predict', imgName_predict)

    imgName_predict_data = sio.loadmat(imgName_predict,verify_compressed_data_integrity=False)
    Iorg_y_predict = imgName_predict_data[predict_save_name] # mat

    # imgName_predict_data = cv2.imread(imgName_predict,0)
    # Iorg_y_predict = imgName_predict_data/255.0  # png

    Iorg_y = sio.loadmat(imgName,verify_compressed_data_integrity=False)
    Iorg_y = Iorg_y[predict_save_name] # mat
    # print(Iorg_y.max())
    # print(Iorg_y_predict.max())
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(Iorg_y)
    # plt.subplot(122)
    # plt.imshow(Iorg_y_predict)
    # plt.show()


    # Img = cv2.imread(imgName, 1)
    # Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    # Img_rec_yuv = Img_yuv.copy()
    # Iorg_y = Img_yuv[:, :, 0]


    if trunc_mode == 'HU':
        Iorg_y_predict = Iorg_y_predict
        Iorg_y = Iorg_y
    elif trunc_mode == 'liver':
        Iorg_y_predict = trunc(denormalize_(Iorg_y_predict, 3072.0, -1024.0),240,-160)
        Iorg_y = trunc(denormalize_(Iorg_y, 3072.0, -1024.0),240,-160)
    elif trunc_mode == 'lung':
        Iorg_y_predict = trunc(denormalize_(Iorg_y_predict, 3072.0, -1024.0), 150, -1350)
        Iorg_y = trunc(denormalize_(Iorg_y, 3072.0, -1024.0), 150, -1350)


    rec_PSNR = psnr(Iorg_y_predict.astype(np.float64)*255, Iorg_y.astype(np.float64)*255)
    rec_SSIM = ssim(Iorg_y_predict.astype(np.float64), Iorg_y.astype(np.float64), data_range= 1)
    m_reg = compute_measure(Iorg_y.astype(np.float64) , Iorg_y_predict, 1)
    RMSE_total.append(m_reg)
    print("[%02d/%02d] Run for %s , PSNR is %.4f, SSIM is %.4f，RMSE is %.4f" % (
    img_no, ImgNum, imgName_predict, rec_PSNR, rec_SSIM,m_reg))

    if save_mode == 1:
        # 保存图像到本地
        im_rec_rgb = np.clip(Iorg_y_predict*255.0, 0, 255).astype(np.uint8)
        # # print(imgName)
        # img_name = imgName_predict.split('\\')
        # # print(img_name)
        # img_name = str(img_name[1])
        # # print(img_name)
        # img_name = img_name.split('.')
        # print(img_name)
        img_dir = predict_image_str_save + str(img_no+1) + "_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (rec_PSNR, rec_SSIM, m_reg)
        # print(img_dir)
        cv2.imwrite(img_dir, im_rec_rgb)

        x_test_all.append(Iorg_y_predict)  # assemble


    PSNR_All[0, img_no] = rec_PSNR
    SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = " Avg Proposed PSNR/SSIM/RMSE is %.3f-%.3f/%.5f-%.5f/%.5f-%.5f \n" % \
              (np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All),
               np.array(RMSE_total).mean(), np.array(RMSE_total).std())
print(output_data)
print("CS Reconstruction End")
# print('PSNR=',np.mean(PSNR_All),'SSIM=',np.mean(SSIM_All))

if save_mode ==1:
    x_test_all = np.array(x_test_all)
    # save_npy_name = result_dir + 'FBP_' + str(self.ds_factor) + '.npy'
    # print('Save test npy file shape = ', x_test_all.shape)
    # np.save(save_npy_name, x_test_all)  # save npy

    # 保存为nii, 可视化
    out = sitk.GetImageFromArray(x_test_all * 4096 - 1024)
    save_npy_name = predict_image_str_save + 'FBP_' + str(ds_factor) + '.nii.gz'
    sitk.WriteImage(out, save_npy_name)

    # save result
    output_file_name = predict_image_str_save + 'result_rmse_all.txt'
    output_data_psnr = [RMSE_total]
    output_file = open(output_file_name, 'a')
    for fp in output_data_psnr:  # write data in txt
        output_file.write(str(fp))
        output_file.write(',')
    output_file.write('\n')  # line feed
    output_file.close()
##############################################################################################################