%% FISTA-TV for sparse view CT reconstruction，FISTA-TV重建的代码
clc;
close all; 
clear;

%% paramters
addpath('./TV'); 
addpath('./npy2matlab');

%% parameters
save_mode = 1 ; % 1 save ,0 donnot save

%%
% full_sampled = load('../../data/CT/zg_window/test/full_sampled.mat'); % load data
full_sampled = load('../../data/CT/HU/test/full_sampled.mat'); % load data
save_dir = ['./result_HU/'];  % reconstruction str
% full_sampled = load('../../data/XIANGTAN/test1/full_sampled.mat'); % load data
% save_dir = ['./result_xiangtan/'];  % reconstruction str

D = 2000 ;  % 扇形射线束的顶点到旋转中心的距离（单位为像素）
S_D = 731/1024 ; % FanSensorSpacing
A_R = 360/1024 ; % 360/1024 FanRotationIncrement

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

    B1_line = fanbeam(Xgt,D,'FanSensorGeometry','line','FanSensorSpacing',S_D,'FanRotationIncrement',A_R);
    f3 = ifanbeam(B1_line,D,'FanSensorGeometry','line','FanSensorSpacing',S_D,'FanRotationIncrement',A_R);
    Xgt_label = imresize(abs(f3), size(Xgt));  % FBP
    
%     error = max(max(abs(X1-Xgt_label)));
%     error_map = abs(X1-Xgt_label);
% %     figure()
% %     imshow(Xgt_label)
% %     figure()
% %     imshow(X1)
%     figure()
%     imshow(error_map)
% %     figure()
% %     imshow(Xgt)
%     figure()
%     imshow(B1_line,[])
    
    if save_mode ==1
        save_dir_label_mat = [save_dir,'label_mat/'];
        if ~exist(save_dir_label_mat,'dir') % creat save file
            mkdir(save_dir_label_mat) 
        end
        im_rec_savename1 = [save_dir_label_mat,num2str(i),'.mat'];
        im_rec = Xgt_label;
        save(im_rec_savename1,'im_rec');
    end
    close all;
end

