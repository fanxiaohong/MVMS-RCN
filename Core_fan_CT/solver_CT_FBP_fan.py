import os
import torch
import torchvision
import torch.nn as nn
from functions import *
import glob
from skimage.measure import compare_ssim as ssim
import cv2
from time import time
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
from torch_radon import Radon, RadonFanbeam
###########################################################################
class Solver_CT_FBP(object):
    def __init__(self, test_loader,theta, args, device):
        self.test_loader = test_loader
        self.theta = theta
        self.device = device
        self.CT_test_name = args.CT_test_name
        self.model_dir = args.model_dir
        self.net_name = args.net_name
        self.log_dir = args.log_dir
        self.result_dir = args.result_dir
        self.ds_factor = args.ds_factor
        self.save_image_mode = args.save_image_mode

        self.theta_label = np.linspace(0, 2 * np.pi, int(1024), endpoint=False)

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        mat_scale = (mat - self.trunc_min)/(self.trunc_max - self.trunc_min) # scale to 0-1
        return mat_scale

    def test(self):
        # initial test file
        result_dir_tmp = os.path.join(self.result_dir, self.CT_test_name)
        result_dir = result_dir_tmp + '_' + self.net_name + '_view_' + str(self.ds_factor) + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with torch.no_grad():
            RMSE_total = []
            PSNR_total = []
            SSIM_total = []
            image_num = 0
            time_all = 0
            x_test_all = []
            for data in self.test_loader:
                image_num = image_num +1
                full_sampled_x = data
                # test torch radon
                full_sampled_x = torch.unsqueeze(full_sampled_x, 1).to(self.device)

                det_count = 1024
                source_distance = 500
                det_spacing = 2

                radon_label = RadonFanbeam(full_sampled_x.shape[2], self.theta_label,
                                           source_distance=source_distance, det_distance=source_distance,
                                           det_count=det_count, det_spacing=det_spacing)
                radon = RadonFanbeam(full_sampled_x.shape[2], self.theta,
                                     source_distance=source_distance, det_distance=source_distance,
                                     det_count=det_count, det_spacing=det_spacing)

                sinogram_label = radon_label.forward(full_sampled_x)
                b_in_label = radon_label.filter_sinogram(sinogram_label)
                batch_x = torch.abs(radon_label.backprojection(b_in_label))

                start = time()
                sinogram = radon.forward(full_sampled_x)
                b_in = radon.filter_sinogram(sinogram)
                x_output = radon.backprojection(b_in)   # FBP
                end = time()
                time_all = time_all + end -start

                batch_x = batch_x.cpu().data.numpy().reshape(batch_x.shape[2], batch_x.shape[3])
                x_output = x_output.cpu().data.numpy().reshape(batch_x.shape[0], batch_x.shape[1])

                # calculate index
                rec_PSNR = psnr(x_output * 255.0, batch_x * 255.0)
                rec_SSIM = ssim(x_output, batch_x, data_range=1)
                rec_RMSE = compute_measure(batch_x, x_output, 1)
                PSNR_total.append(rec_PSNR)
                SSIM_total.append(rec_SSIM)
                RMSE_total.append(rec_RMSE)

                # save image
                if self.save_image_mode == 1:
                    img_dir = result_dir + str(image_num) + "_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
                        rec_PSNR, rec_SSIM, rec_RMSE)
                    print(img_dir)
                    im_rec_rgb = np.clip(x_output * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(img_dir, im_rec_rgb)
                    print("Mean run time for %s test is %.6f, Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (
                        image_num, (end - start), rec_PSNR, rec_SSIM, rec_RMSE))
                elif self.save_image_mode == 2:
                    x_test_all.append(x_output)  # assemble

            if self.save_image_mode == 2:
                x_test_all = np.array(x_test_all)
                save_npy_name = result_dir + 'FBP_' + str(self.ds_factor) + '.npy'
                print('Save test npy file shape = ', x_test_all.shape)
                np.save(save_npy_name, x_test_all)  # save npy

                # 保存为nii, 可视化
                out = sitk.GetImageFromArray(x_test_all * 4096 - 1024)
                save_npy_name = result_dir + 'FBP_' + str(self.ds_factor) + '.nii.gz'
                sitk.WriteImage(out, save_npy_name)

            PSNR_mean = np.array(PSNR_total).mean()
            PSNR_std = np.array(PSNR_total).std()
            SSIM_mean = np.array(SSIM_total).mean()
            SSIM_std = np.array(SSIM_total).std()
            RMSE_mean = np.array(RMSE_total).mean()
            RMSE_std = np.array(RMSE_total).std()
            print("Mean run time for test is %.6f, Proposed PSNR is %.3f-%.3f, SSIM is %.5f-%.5f, RMSE is %.5f-%.5f" % (
                time_all / image_num, PSNR_mean,PSNR_std, SSIM_mean,SSIM_std,RMSE_mean,RMSE_std))

        # save result
        output_file_name1 = result_dir + 'result_rmse_all.txt'
        output_data_psnr = [RMSE_total]
        output_file1 = open(output_file_name1, 'a')
        for fp in output_data_psnr:  # write data in txt
            output_file1.write(str(fp))
            output_file1.write(',')
        output_file1.write('\n')  # line feed
        output_file1.close()

        output_data = [PSNR_mean,PSNR_std,SSIM_mean,SSIM_std,RMSE_mean,RMSE_std]
        output_file_name = result_dir_tmp + '_' + self.net_name + '_ds_' + str(self.ds_factor)+ ".txt"
        output_file = open(output_file_name, 'a')
        for fp in output_data:  # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')  # line feed
        output_file.close()


