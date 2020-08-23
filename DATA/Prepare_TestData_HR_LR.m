function Prepare_TestData_HR_LR()
clear all; close all; clc
path_original = './benchmark/';
% dataset  = {'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'};
dataset  = {'Set5'};
ext = {'*.jpg', '*.png', '*.bmp'};

degradation = 'bicubic'; 
scale_all = [2,3,4];


for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, 'HR', ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        im_ori = imread(fullfile(path_original, dataset{idx_set}, 'HR', name_im));
        if size(im_ori, 3) == 1
            im_ori = cat(3, im_ori, im_ori, im_ori);
        end
        for scale = scale_all
            fprintf('x%d ', scale);
            im_HR = modcrop(im_ori, scale);
            im_LR = imresize(im_HR, 1/scale, 'bicubic');
            % folder
            folder_LR = fullfile([path_original, dataset{idx_set} '/LR_', degradation], ['X', num2str(scale)]);
            if ~exist(folder_LR)
                mkdir(folder_LR)
            end
            % fn
            fn_LR = fullfile(folder_LR, [name_im(1:end-4), 'x', num2str(scale), '.png']);
            imwrite(im_LR, fn_LR, 'png');
        end
        fprintf('\n');
    end
    fprintf('\n');
end
end
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end













