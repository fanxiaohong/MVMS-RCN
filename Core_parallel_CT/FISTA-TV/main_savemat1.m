%% FISTA-TV for sparse view CT reconstruction，FISTA-TV重建的代码
clc;
close all; 
clear;

%% paramters
addpath('./TV'); 
addpath('./npy2matlab');

%% parameters
ds_factor_all = [12,8,6,4];   % sparse view ,48/15,24/30,12/60,8/90,6/120,4/180
Fista_tv_lambda_all = [0.65,0.3,0.2,0.05] ; % 0.05， FISTA-TV regularization cofficient
% ds_factor_all = [4];   % sparse view ,48/15,24/30,12/60,8/90,6/120,4/180
% Fista_tv_lambda_all = [0.08] ; % 0.05， FISTA-TV regularization cofficient
% ds_factor_all = [4];   % sparse view ,48/15,24/30,12/60,8/90,6/120,4/180
% Fista_tv_lambda_all = [0.15] ; % 0.05， FISTA-TV regularization cofficient
Fista_tv_iter = 100 ; % 200 default,
save_mode = 1 ; % 1 save ,0 donnot save

%%
time_FBP_all_ds = [] ;
time_FISTA_TV_all_ds = [];
for iii = 1 : length(ds_factor_all)
    Fista_tv_lambda = Fista_tv_lambda_all(iii);
    ds_factor = ds_factor_all(iii) ;
%%
% sample = readNPY(['../../model and result CT/FBP/CT_test_CT-FBP_ds_',num2str(ds_factor),'/FBP_',num2str(ds_factor),'.npy']);
full_sampled = load('../../data/CT/HU/test_dataset2/full_sampled.mat'); % load data
% full_sampled = load('../../data/XIANGTAN/test1/full_sampled.mat'); % load data
save_dir = ['./result_HU/ds',num2str(ds_factor),'_dataset2/'];  % reconstruction str
Np = 720;


%%
if save_mode == 1
    if ~exist(save_dir,'dir')
        mkdir(save_dir)
    end
end
%% main circulate
full_sampled_image = full_sampled.image_all;
data_size = size(full_sampled_image);
time_FBP = []; PSNR_all_FBP=[];
time_FISTA_TV = []; PSNR_all_FISTA_TV=[];
for i = 1:data_size(1)
    i % print image num
    Xgt = squeeze(full_sampled_image(i,:,:));  % Ground Truth
%     Xgt_label = squeeze(label_image(i,:,:));
%     Xgt_label = (Xgt_label-Xgt_min)/(Xgt_max-Xgt_min); % scale to 0-1
    img_size = size(Xgt);

    % generate projection data from phantom using radon transform
    theta = (0:Np-1)*180/Np;
    y = radon(Xgt, theta);    % 729 pixels; 720 projection views; Mayo CT
    Xgt_label = iradon(y,theta);
    Xgt_label = imresize(abs(Xgt_label), size(Xgt));  % FBP

    % downsample projection data from full-view CT
    y_ds = y(:,1:ds_factor:end);
    theta_ds = theta(1:ds_factor:end);
    tic
    X_0 = iradon(y_ds,theta_ds);
    X_0 = imresize(X_0, size(Xgt));  % FBP
    CPU_time_FBP = toc;
    time_FBP  = [time_FBP ,CPU_time_FBP];
    PSNR_image_FBP = PSNR(Xgt_label*255, abs(X_0)*255.0); % caculate PSNR
    PSNR_all_FBP = [PSNR_all_FBP,PSNR_image_FBP]; 
    
    % Iterative Total Varaition using FISTA
    % 'iso'--> isotropic TV
    % 'l1' --> l1-based, anisotropic TV
    pars.tv = 'iso';   % 'iso'--> isotropic TV
    pars.MAXITER = Fista_tv_iter;
    pars.fig = 0; % 1,figure ,0 donnot
    tic
    X_fista_tv = tv_fista(y_ds,theta_ds,img_size, Fista_tv_lambda,-Inf,Inf,pars); % FISTA-TV,0.05
    CPU_time_FISTA_TV=toc;
    time_FISTA_TV  = [time_FISTA_TV ,CPU_time_FISTA_TV];
    PSNR_image_FISTA_TV = PSNR(Xgt_label*255,  abs(X_fista_tv)*255.0); % caculate PSNR
    PSNR_all_FISTA_TV = [PSNR_all_FISTA_TV,PSNR_image_FISTA_TV]; 
    if save_mode ==1
        save_dir_FISTA_TV_png = [save_dir,num2str(Fista_tv_iter),'-',num2str(Fista_tv_lambda),'_png/'];
        save_dir_FISTA_TV_mat = [save_dir,num2str(Fista_tv_iter),'-',num2str(Fista_tv_lambda),'_mat/'];
        if ~exist(save_dir_FISTA_TV_png,'dir') % creat save file
            mkdir(save_dir_FISTA_TV_png) 
        end
         if ~exist(save_dir_FISTA_TV_mat,'dir') % creat save file
            mkdir(save_dir_FISTA_TV_mat) 
        end
        im_rec_savename1 = [save_dir_FISTA_TV_mat,num2str(i),'.mat'];
        im_rec = abs(X_fista_tv);
        save(im_rec_savename1,'im_rec');
        im_rec_savename = [save_dir_FISTA_TV_png,num2str(i),'.png'];
        imwrite(im_rec,im_rec_savename);
    end
    close all;
end

%% show reconstruction result
time_FBP_mean = mean(time_FBP(:))
time_FISTA_TV_mean = mean(time_FISTA_TV(:))
PSNR_all_FBP_mean = mean(PSNR_all_FBP(:))
PSNR_all_FISTA_TV_mean = mean(PSNR_all_FISTA_TV(:))
time_FBP_all_ds = [time_FBP_all_ds,time_FBP_mean] ;
time_FISTA_TV_all_ds = [time_FISTA_TV_all_ds,time_FISTA_TV_mean ];
end

time_FBP_all_ds_mean = mean(time_FBP_all_ds) 
time_FISTA_TV_all_ds_mean = mean(time_FISTA_TV_all_ds)
PSNR_all_FISTA_TV_mean = mean(PSNR_all_FISTA_TV(:))
