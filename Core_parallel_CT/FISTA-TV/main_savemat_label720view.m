%% FISTA-TV for sparse view CT reconstruction，FISTA-TV重建的代码
clc;
close all; 
clear;

% 运行保存label图像数据

%% paramters
addpath('./TV'); 
addpath('./npy2matlab');

%% parameters
save_mode = 1 ; % 1 save ,0 donnot save

%%
% full_sampled = load('../../data/CT/zg_window/test/full_sampled.mat'); % load data
full_sampled = load('../../data/CT/HU/test_dataset2/full_sampled.mat'); % load data
save_dir = ['./result_HU/'];  % reconstruction str
% full_sampled = load('../../data/XIANGTAN/test1/full_sampled.mat'); % load data
% save_dir = ['./result_xiangtan/'];  % reconstruction str

Np = 720;

%
if save_mode == 1
    if ~exist(save_dir,'dir')
        mkdir(save_dir)
    end
end
%% main circulate
full_sampled_image = full_sampled.image_all;
data_size = size(full_sampled_image);
for i = 1:data_size(1)
    i % print image num
    Xgt = squeeze(full_sampled_image(i,:,:));  % Ground Truth
    img_size = size(Xgt);

    % generate projection data from phantom using radon transform
    theta = (0:Np-1)*180/Np;
    y = radon(Xgt, theta);    % 729 pixels; 720 projection views; Mayo CT
    Xgt_label = iradon(y,theta);
    Xgt_label = imresize(abs(Xgt_label), size(Xgt));  % FBP
    
    if save_mode ==1
        save_dir_label_mat = [save_dir,'label_mat_dataset2/'];
        if ~exist(save_dir_label_mat,'dir') % creat save file
            mkdir(save_dir_label_mat) 
        end
        im_rec_savename1 = [save_dir_label_mat,num2str(i),'.mat'];
        im_rec = Xgt_label;
        save(im_rec_savename1,'im_rec');
    end
    close all;
end

