%% This has never been done before
clear;clc;

% Parameter configuration
tic
mode = 'train'; % val
if strcmp(mode,'train')
    startnum = 1;
    endnum = 600; 
end

if strcmp(mode,'val')
    startnum = 301;
    endnum = 350;
end

datafolder = 'HeavyRainSynthetic';
x = 1;
y = 1;
dx = 720-1;
dy = 480-1;
if ~exist('train', 'file')
    mkdir('train/in/');
    mkdir('train/atm/');
    mkdir('train/trans/');
    mkdir('train/streak/');
    mkdir('train/gt/');
end
if ~exist('val', 'file')
    mkdir('val/in/');
    mkdir('val/atm/');
    mkdir('val/trans/');
    mkdir('val/streak/');
    mkdir('val/gt/');
end

if ~exist('filelists', 'file')
    mkdir('filelists/')
end

f_in = fopen(sprintf('filelists/%s_in.txt', mode), 'w');
f_st = fopen(sprintf('filelists/%s_streak.txt', mode), 'w'); 
f_trans = fopen(sprintf('filelists/%s_trans.txt', mode), 'w');
f_atm = fopen(sprintf('filelists/%s_atm.txt', mode), 'w'); 
f_clean = fopen(sprintf('filelists/%s_clean.txt', mode), 'w'); 

%% Render Image

% Set up directory
root_dir = '/home/liruoteng/Documents/MATLAB/B11-CleanNUS';
image_dir = [root_dir, filesep, 'CleanCollection2', ];
depth_dir = [root_dir , filesep , 'Depth2'];

image_files = dir([image_dir, filesep, '*.png']); 
depth_files = dir([depth_dir, filesep, '*.png']); 
num_of_files = length(image_files); 
counter = 1; 
% Render each image
for i = startnum:endnum
    fileindex= i;
    imname = image_files(i).name;
    depname = depth_files(i).name;
%     assert(strcmp(imname(1:end-4),depname(1:end-9)));
    
    % read image
    img =im2double(imread([image_files(i).folder, filesep, imname]));
    
    depth_img = im2double(rgb2gray(imread([depth_files(i).folder, filesep, depname]))); 
    
    % inverse normalize depth map
    dep = 1./(depth_img + 0.1);
    dep = dep / max(dep(:)); 
    
    imwrite(img(y:y+dy, x:x+dx, :), sprintf('%s/gt/im_%04d.png', mode, fileindex))
    
    for s = 1:5
        theta = s * 5 + 75;
        for atmlevel = 4:6
            tic           
            %% Render Streak
            seed = min(1, abs(normrnd(0.5,0.5)));
            im = imgaussfilt(img, seed);
            
            [rain, streak] = render_rain(im, theta, 0.1, -4, 0.7);
            
            %% Render Haze
            [haze, trans, atm] = render_haze(rain, dep); 

            %% Crop Image
            haze = haze(y:y+dy, x:x+dx, :); 
            trans = trans(y:y+dy, x:x+dx, :); 
            atm = atm(y:y+dy, x:x+dx, :); 
            rain = rain(y:y+dy, x:x+dx, :); 
            streak = streak(y:y+dy, x:x+dx, :); 
            
            
            % ======= TO REMOVE ==========
            diff = (haze - (1-trans) .* atm )./trans - streak - im(y:y+dy, x:x+dx, :);
            if diff > 0.0001
                fprintf('%f, %f, %f', i, s, atmlevel);
            end
            %imshow(haze);
            %% Write to File
            
            imwrite((haze), sprintf('%s/in/im_%04d_s%02d_a%02d.png',mode, fileindex, theta, atmlevel));
            imwrite((streak), sprintf('%s/streak/im_%04d_s%02d_a%02d.png', mode, fileindex, theta, atmlevel));
            imwrite((trans), sprintf('%s/trans/im_%04d_s%02d_a%02d.png', mode, fileindex, theta, atmlevel));
            imwrite((atm), sprintf('%s/atm/im_%04d_s%02d_a%02d.png', mode, fileindex, theta, atmlevel)); 
%             imwrite(im(y:y+dy, x:x+dx, :), sprintf('%s/gt/im_%04d.png', mode, fileindex))

            fprintf(f_in,  sprintf('../../../data/%s/%s/in/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
            fprintf(f_trans, sprintf('../../../data/%s/%s/trans/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
            fprintf(f_atm, sprintf('../../../data/%s/%s/atm/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
            fprintf(f_st, sprintf('../../../data/%s/%s/streak/im_%04d_s%02d_a%02d.png\n', datafolder,mode, fileindex, theta, atmlevel)); 
            fprintf(f_clean, sprintf('../../../data/%s/%s/gt/im_%04d.png\n', datafolder, mode, fileindex));
            
            fprintf('Num: %d, time elapsed: %f, sigma: %f\n', counter, toc, seed); 
            counter = counter + 1; 
        end
    end
end

toc

% 
% gaussdep = imgaussfilt(dep, 100); 
% 
% imshow(gaussdep); 
% beta = 1;
% 
% txmap = exp(-beta * gaussdep); 
% 
% gaussdepdisp = imresize(gaussdep, 0.25); 
% txmapdisp = imresize(txmap, 0.25); 
% 
% imshow([gaussdepdisp; txmapdisp]); 
% 
% fprintf('max: %f, min: %f', max(txmap(:)), min(txmap(:))); 
