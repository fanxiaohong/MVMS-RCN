import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import RED_CNN
from measure import compute_measure
import cv2
from time import time
import SimpleITK as sitk


class Solver(object):
    def __init__(self, args, data_loader,devicer):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.ds_factor = args.ds_factor
        self.device =devicer
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.trunc_mode = args.trunc_mode
        self.save_path = args.save_path
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_epoch = args.decay_epoch
        self.save_epoch = args.save_epoch
        self.test_epoch = args.test_epoch
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.REDCNN = RED_CNN()
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, epoch):
        f = os.path.join(self.save_path, 'REDCNN_{}epoch.pkl'.format(epoch))
        torch.save(self.REDCNN.state_dict(), f)


    def load_model(self, epoch):
        f = os.path.join(self.save_path, 'REDCNN_{}epoch.pkl'.format(epoch))
        self.REDCNN.load_state_dict(torch.load(f))

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        mat_scale = (mat - self.trunc_min)/(self.trunc_max - self.trunc_min) # scale to 0-1
        return mat_scale


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result,ds_factor):
        pred = pred.numpy()
        rec_PSNR=pred_result[0]
        rec_SSIM=pred_result[1]
        rec_RMSE=pred_result[2]

        img_dir = self.save_path +'ds'+ str(ds_factor)+'_fig/'+ str(fig_name+1) +"_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
            rec_PSNR, rec_SSIM, rec_RMSE)
        print(img_dir)
        cv2.imwrite(img_dir, pred*255.0)
        print("Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (rec_PSNR, rec_SSIM, rec_RMSE))

    def train(self):
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.REDCNN.parameters())))

        train_losses = []
        total_iters = 0
        # start_time = time.time()
        for epoch in range(self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}".format(total_iters, epoch,
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item()))
            # learning rate decay
            if (epoch+1) % self.decay_epoch == 0:
                self.lr_decay()
            # save model
            if (epoch+1) % self.save_epoch == 0:
                self.save_model(epoch+1)
                np.save(os.path.join(self.save_path, 'loss_{}_epoch.npy'.format(epoch+1)), np.array(train_losses))


    def test(self):
        del self.REDCNN
        # load
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_epoch)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = [], [], []
        time_all = 0
        image_num = 0

        x_test_all = []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                image_num = image_num + 1
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                start = time()
                pred = self.REDCNN(x)
                end = time()
                time_all = time_all + end - start

                # denormalize, truncate
                x = x.view(shape_, shape_).cpu().detach()
                y = y.view(shape_, shape_).cpu().detach()
                pred = pred.view(shape_, shape_).cpu().detach()

                # HU transform
                if self.trunc_mode == 'HU':
                    x = x
                    y = y
                    pred = pred
                else:  # liver, lung
                    x = self.trunc(self.denormalize_(x))
                    y = self.trunc(self.denormalize_(y))
                    pred = self.trunc(self.denormalize_(pred))

                data_range = 1

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg.append(pred_result[0])
                pred_ssim_avg.append(pred_result[1])
                pred_rmse_avg.append(pred_result[2])

                if self.mode == 'test':
                    pred_cpu = pred.cpu().data.numpy().reshape(pred.shape[1], pred.shape[1])
                    x_test_all.append(pred_cpu)  # assemble

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result,self.ds_factor)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)

            if self.mode == 'test':
                x_test_all = np.array(x_test_all)

                # 保存为nii, 可视化
                out = sitk.GetImageFromArray(x_test_all * 4096 - 1024)
                result_dir = self.save_path + 'ds' + str(self.ds_factor) + '_fig/'
                save_npy_name = result_dir + 'FBP_' + str(self.ds_factor) + '.nii.gz'
                sitk.WriteImage(out, save_npy_name)

            print('\n')
            print('Original === \nPSNR avg: {:.3f} \nSSIM avg: {:.3f} \nRMSE avg: {:.3f}'.format(ori_psnr_avg/len(self.data_loader),
                                                                                            ori_ssim_avg/len(self.data_loader),
                                                                                            ori_rmse_avg/len(self.data_loader)))
            PSNR_mean = np.array(pred_psnr_avg).mean()
            PSNR_std = np.array(pred_psnr_avg).std()
            SSIM_mean = np.array(pred_ssim_avg).mean()
            SSIM_std = np.array(pred_ssim_avg).std()
            RMSE_mean = np.array(pred_rmse_avg).mean()
            RMSE_std = np.array(pred_rmse_avg).std()
            print("ds factor is %.0f, Mean run time for test is %.6f,"
                  " PSNR is %.3f-%.3f, SSIM is %.5f-%.5f, RMSE is %.5f-%.5f" % (
                      self.ds_factor, time_all / image_num, PSNR_mean, PSNR_std,
                      SSIM_mean, SSIM_std, RMSE_mean, RMSE_std))

            # save result
            output_file_name1 =  self.save_path + 'ds' + str(self.ds_factor) + '_fig/' + 'result_rmse_all.txt'
            output_data_rmse = [pred_rmse_avg]
            output_file1 = open(output_file_name1, 'a')
            for fp in output_data_rmse:  # write data in txt
                output_file1.write(str(fp))
                output_file1.write(',')
            output_file1.write('\n')  # line feed
            output_file1.close()