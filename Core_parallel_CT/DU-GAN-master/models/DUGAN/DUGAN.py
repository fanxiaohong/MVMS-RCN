import os
from torch.nn import functional as F
import torch
import numpy as np
import copy
import torchvision
import argparse
import tqdm
import torch.nn as nn

from models.basic_template import TrainTask
from utils.grad_loss import SobelOperator
from .DUGAN_wrapper import UNet
from models.REDCNN.REDCNN_wrapper import Generator
from utils.gan_loss import ls_gan
from utils.ops import turn_on_spectral_norm
# from utils.metrics import compute_ssim, compute_psnr, compute_rmse
from skimage.measure import compare_ssim as ssim
from functions import *
from time import time
import cv2
import SimpleITK as sitk

'''
python main.py --batch_size=64 --cr_loss_weight=5.08720932695335 --cutmix_prob=0.7615524094697519 --cutmix_warmup_iter=1000 --d_lr=7.122979672016055e-05 --g_lr=0.00018083340390609657 --grad_gen_loss_weight=0.11960717521104237 --grad_loss_weight=35.310016043755894 --img_gen_loss_weight=0.14178356036938378 --max_iter=50000 --model_name=UnetGAN --num_channels=32 --num_layers=10 --num_workers=32 --pix_loss_weight=5.034293425614828 --print_freq=10 --run_name=newest --save_freq=2500 --test_batch_size=8 --test_dataset_name=lmayo_test_512 --train_dataset_name=lmayo_train_64 --use_grad_discriminator=true --weight_decay 0. --num_workers 4
python main.py --batch_size=64 --cr_loss_weight=5.08720932695335 --cutmix_prob=0.7615524094697519 --cutmix_warmup_iter=1000 --d_lr=7.122979672016055e-05 --g_lr=0.00018083340390609657 --grad_gen_loss_weight=0.11960717521104237 --grad_loss_weight=35.310016043755894 --img_gen_loss_weight=0.14178356036938378 --max_iter=100000 --model_name=UnetGAN --num_channels=32 --num_layers=10 --num_workers=32 --pix_loss_weight=5.034293425614828 --print_freq=10 --run_name=newest --save_freq=2500 --test_batch_size=8 --test_dataset_name=cmayo_test_512 --train_dataset_name=cmayo_train_64 --use_grad_discriminator=true --weight_decay 0. --num_workers 4
'''


class DUGAN(TrainTask):

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--num_layers", default=10, type=int)
        parser.add_argument("--num_channels", default=32, type=int)
        # Need D conv_dim 64
        parser.add_argument("--g_lr", default=1e-4, type=float)
        parser.add_argument("--d_lr", default=1e-4, type=float)
        parser.add_argument("--d_iter", default=1, type=int)
        parser.add_argument("--cutmix_prob", default=0.5, type=float)
        parser.add_argument("--img_gen_loss_weight", default=0.1, type=float)
        parser.add_argument("--grad_gen_loss_weight", default=0.1, type=float)
        parser.add_argument("--pix_loss_weight", default=1., type=float)
        parser.add_argument("--grad_loss_weight", default=20., type=float)
        parser.add_argument("--cr_loss_weight", default=1.0, type=float)
        parser.add_argument("--cutmix_warmup_iter", default=1000, type=int)
        parser.add_argument("--use_grad_discriminator", help='use_grad_discriminator', type=bool, default=True)
        parser.add_argument("--moving_average", default=0.999, type=float)
        parser.add_argument("--repeat_num", default=6, type=int)
        return parser

    def set_model(self):
        opt = self.opt
        generator = Generator(in_channels=1, out_channels=opt.num_channels, num_layers=opt.num_layers, kernel_size=3,
                              padding=1)
        g_optimizer = torch.optim.Adam(generator.parameters(), opt.g_lr, weight_decay=opt.weight_decay)

        self.gan_metric = ls_gan
        img_discriminator = UNet(repeat_num=opt.repeat_num, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        img_discriminator = turn_on_spectral_norm(img_discriminator)
        img_d_optimizer = torch.optim.Adam(img_discriminator.parameters(), opt.d_lr)
        grad_discriminator = copy.deepcopy(img_discriminator)
        grad_d_optimizer = torch.optim.Adam(grad_discriminator.parameters(), opt.d_lr)

        ema_generator = copy.deepcopy(generator)

        self.logger.modules = [generator, g_optimizer, img_discriminator, img_d_optimizer, grad_discriminator,
                               grad_d_optimizer, ema_generator]

        self.sobel = SobelOperator().cuda()
        self.generator = generator.cuda()
        self.g_optimizer = g_optimizer
        self.img_discriminator = img_discriminator.cuda()
        self.img_d_optimizer = img_d_optimizer
        self.grad_discriminator = grad_discriminator.cuda()

        self.ema_generator = ema_generator.cuda()

        self.grad_d_optimizer = grad_d_optimizer
        self.apply_cutmix_prob = torch.rand(opt.max_iter)

    def train_discriminator(self, discriminator, d_optimizer,
                            full_dose, low_dose, gen_full_dose, prefix, n_iter=0):
        opt = self.opt
        msg_dict = {}
        ############## Train Discriminator ###################
        d_optimizer.zero_grad()
        real_enc, real_dec = discriminator(full_dose)
        fake_enc, fake_dec = discriminator(gen_full_dose.detach())
        source_enc, source_dec = discriminator(low_dose)
        msg_dict.update({
            'enc/{}_real'.format(prefix): real_enc,
            'enc/{}_fake'.format(prefix): fake_enc,
            'enc/{}_source'.format(prefix): source_enc,
            'dec/{}_real'.format(prefix): real_dec,
            'dec/{}_fake'.format(prefix): fake_dec,
            'dec/{}_source'.format(prefix): source_dec,
        })

        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                    self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                    self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)
        total_loss = disc_loss

        apply_cutmix = self.apply_cutmix_prob[n_iter - 1] < warmup(opt.cutmix_warmup_iter, opt.cutmix_prob, n_iter)
        if apply_cutmix:
            mask = cutmix(real_dec.size()).to(real_dec)

            # if random.random() > 0.5:
            #     mask = 1 - mask

            cutmix_enc, cutmix_dec = discriminator(mask_src_tgt(full_dose, gen_full_dose.detach(), mask))

            cutmix_disc_loss = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)

            cr_loss = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))

            total_loss += cutmix_disc_loss + cr_loss * opt.cr_loss_weight

            msg_dict.update({
                'enc/{}_cutmix'.format(prefix): cutmix_enc,
                'dec/{}_cutmix'.format(prefix): cutmix_dec,
                'loss/{}_cutmix_disc'.format(prefix): cutmix_disc_loss,
                'loss/{}_cr'.format(prefix): cr_loss,
            })

        total_loss.backward()

        d_optimizer.step()
        # self.logger.msg(msg_dict, n_iter)

    def update_moving_average(self):
        opt = self.opt
        m = opt.moving_average
        for old_param, new_param in zip(self.ema_generator.parameters(), self.generator.parameters()):
            old_param.data = old_param.data * m + new_param.data * (1. - m)

    def train(self, inputs, n_iter):
        opt = self.opt

        self.update_moving_average()

        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

        self.generator.train()
        self.img_discriminator.train()
        self.grad_discriminator.train()
        msg_dict = {}

        # # 打印参数
        # print("Total paramerters 1 is {}  ".format(sum(x.numel() for x in self.generator.parameters())))
        # print("Total paramerters 2 is {}  ".format(sum(x.numel() for x in self.img_discriminator.parameters())))
        # print("Total paramerters 3 is {}  ".format(sum(x.numel() for x in self.grad_discriminator.parameters())))


        gen_full_dose = self.generator(low_dose)
        grad_gen_full_dose = self.sobel(gen_full_dose)
        grad_low_dose = self.sobel(low_dose)
        grad_full_dose = self.sobel(full_dose)
        self.train_discriminator(self.img_discriminator, self.img_d_optimizer,
                                 full_dose, low_dose, gen_full_dose, prefix='img', n_iter=n_iter)

        if n_iter % opt.d_iter == 0:
            ############## Train Generator ###################

            ########### GAN Loss ############
            self.g_optimizer.zero_grad()
            img_gen_enc, img_gen_dec = self.img_discriminator(gen_full_dose)
            img_gen_loss = self.gan_metric(img_gen_enc, 1.) + self.gan_metric(img_gen_dec, 1.)

            total_loss = 0.
            if opt.use_grad_discriminator:
                self.train_discriminator(self.grad_discriminator, self.grad_d_optimizer,
                                         grad_full_dose, grad_low_dose, grad_gen_full_dose, prefix='grad',
                                         n_iter=n_iter)
                grad_gen_enc, grad_gen_dec = self.grad_discriminator(grad_gen_full_dose)
                grad_gen_loss = self.gan_metric(grad_gen_enc, 1.) + self.gan_metric(grad_gen_dec, 1.)
                total_loss = grad_gen_loss * opt.grad_gen_loss_weight
                msg_dict.update({
                    'enc/grad_gen_enc': grad_gen_enc,
                    'dec/grad_gen_dec': grad_gen_dec,
                    'loss/grad_gen_loss': grad_gen_loss,
                })

            ########### Pixel Loss ############
            pix_loss = F.mse_loss(gen_full_dose, full_dose)

            ########### L1 Loss ############
            l1_loss = F.l1_loss(gen_full_dose, full_dose)

            ########### Grad Loss ############
            grad_loss = F.l1_loss(grad_gen_full_dose, grad_full_dose)

            total_loss += img_gen_loss * opt.img_gen_loss_weight + \
                          pix_loss * opt.pix_loss_weight + \
                          grad_loss * opt.grad_loss_weight

            total_loss.backward()

            self.g_optimizer.step()
            msg_dict.update({
                'enc/img_gen_enc': img_gen_enc,
                'dec/img_gen_dec': img_gen_dec,
                'loss/img_gen_loss': img_gen_loss,
                'loss/pix': pix_loss,
                'loss/l1': l1_loss,
                'loss/grad': grad_loss,
            })

            if  n_iter % 1000 == 0 :
                self.logger.msg(msg_dict, n_iter)

    @torch.no_grad()
    def generate_images(self, n_iter):
        self.generator.eval()
        low_dose, full_dose = self.test_images
        bs, ch, w, h = low_dose.size()
        gen_full_dose = self.generator(low_dose).clamp(0., 1.)
        fake_imgs = [low_dose, full_dose, gen_full_dose,
                     self.img_discriminator(gen_full_dose)[1].clamp(0., 1.),
                     self.grad_discriminator(self.sobel(gen_full_dose))[1].clamp(0., 1.)]
        fake_imgs = torch.stack(fake_imgs).transpose(1, 0).reshape((-1, ch, w, h))
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=5), n_iter, 'test')

    @torch.no_grad()
    def test(self, n_iter,save_dir, run_mode,save_image_mode,trunc_mode):
        self.generator.eval()
        self.ema_generator.eval()
        if  run_mode == 'train':
            gene_List = [self.ema_generator, self.generator]
        elif run_mode =='test':
            gene_List = [self.ema_generator]
        for name, generator in zip(['ema_', ''], gene_List):
            # psnr_score, ssim_score, rmse_score, total_num = 0., 0., 0., 0
            RMSE_total = []
            PSNR_total = []
            SSIM_total = []
            PSNR_FBP_total = []
            image_num = 0
            time_all = 0
            x_test_all = []
            for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
                image_num = image_num + 1
                # batch_size = low_dose.size(0)
                low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
                start = time()
                gen_full_dose = generator(low_dose).clamp(0., 1.)
                end = time()
                time_all = time_all + end - start

                full_dose = full_dose.cpu().data.numpy().reshape(full_dose.shape[2], full_dose.shape[3])
                gen_full_dose = gen_full_dose.cpu().data.numpy().reshape(full_dose.shape[0], full_dose.shape[1])
                low_dose = low_dose.cpu().data.numpy().reshape(full_dose.shape[0], full_dose.shape[1])
                rec_PSNR = psnr(gen_full_dose * 255.0, full_dose * 255.0)
                rec_PSNR_FBP = psnr(low_dose * 255.0, full_dose * 255.0)
                rec_SSIM = ssim(gen_full_dose, full_dose, data_range=1)
                rec_RMSE = compute_measure(full_dose, gen_full_dose, 1)
                PSNR_total.append(rec_PSNR)
                PSNR_FBP_total.append(rec_PSNR_FBP)
                SSIM_total.append(rec_SSIM)
                RMSE_total.append(rec_RMSE)

            #########################################################
                if run_mode == 'test':
                    # save image
                    if save_image_mode == 1:
                        save_dir_img =  save_dir+ '/img_'+ trunc_mode + '/'
                        if not os.path.exists(save_dir_img):
                            os.makedirs(save_dir_img)
                        img_dir = save_dir_img + str(image_num) + "_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
                            rec_PSNR, rec_SSIM, rec_RMSE)
                        print(img_dir)
                        im_rec_rgb = np.clip(gen_full_dose * 255, 0, 255).astype(np.uint8)
                        cv2.imwrite(img_dir, im_rec_rgb)
                        print(" Mean run time for %s test is %.6f, Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (
                                image_num, (end - start), rec_PSNR, rec_SSIM, rec_RMSE))

                    elif save_image_mode == 2:
                        x_test_all.append(gen_full_dose)  # assemble

            if run_mode == 'test':
                if save_image_mode == 2:
                    x_test_all = np.array(x_test_all)
                    # save_npy_name = save_dir + '/FBP_' + str(self.ds_factor) + '.npy'
                    # print('Save test npy file shape = ', x_test_all.shape)
                    # np.save(save_npy_name, x_test_all)  # save npy

                    # 保存为nii, 可视化
                    out = sitk.GetImageFromArray(x_test_all * 4096 - 1024)
                    save_npy_name = save_dir + 'FBP_.nii.gz'
                    sitk.WriteImage(out, save_npy_name)
            ##################################################################

            PSNR_FBP_mean = np.array(PSNR_FBP_total).mean()
            PSNR_FBP_std = np.array(PSNR_FBP_total).std()
            PSNR_mean = np.array(PSNR_total).mean()
            PSNR_std = np.array(PSNR_total).std()
            SSIM_mean = np.array(SSIM_total).mean()
            SSIM_std = np.array(SSIM_total).std()
            RMSE_mean = np.array(RMSE_total).mean()
            RMSE_std = np.array(RMSE_total).std()

            print("Step is %.0f,Mean run time for test is %.6f, Proposed FBP PSNR is %.3f-%.3f,"
                  " PSNR is %.3f-%.3f, SSIM is %.5f-%.5f, RMSE is %.5f-%.5f" %
                  (n_iter, time_all / image_num, PSNR_FBP_mean, PSNR_FBP_std,PSNR_mean, PSNR_std,
                   SSIM_mean, SSIM_std, RMSE_mean, RMSE_std))

            # save result
            output_data = [n_iter, PSNR_FBP_mean, PSNR_FBP_std, PSNR_mean, PSNR_std,
                           SSIM_mean, SSIM_std, RMSE_mean, RMSE_std]
            output_file_name = str(save_dir) + '/DUGAN_' + str(n_iter) + ".txt"
            output_file = open(output_file_name, 'a')
            for fp in output_data:  # write data in txt
                output_file.write(str(fp))
                output_file.write(',')
            output_file.write('\n')  # line feed
            output_file.close()

            if run_mode == 'test':
                if trunc_mode == 'HU':
                    # save result 所有的PSNR值，话小提琴图
                    output_file_name1 = save_dir+'/DUGAN_' + str(n_iter) + trunc_mode+ '_rmse_all.txt'
                    output_data_psnr = [RMSE_total]
                    output_file1 = open(output_file_name1, 'a')
                    for fp in output_data_psnr:  # write data in txt
                        output_file1.write(str(fp))
                        output_file1.write(',')
                    output_file1.write('\n')  # line feed
                    output_file1.close()

def warmup(warmup_iter, cutmix_prob, n_iter):
    return min(n_iter * cutmix_prob / warmup_iter, cutmix_prob)


def cutmix(mask_size):
    mask = torch.ones(mask_size)
    lam = np.random.beta(1., 1.)
    _, _, height, width = mask_size
    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))
    mask[:, :, y0:y1, x0:x1] = 0
    return mask


def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target
