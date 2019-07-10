function [image_rain, actual_streak] = render_rain(img,  theta, ps, density, intensity)

image_rain = img;
h = size(img, 1); 
w = size(img, 2); 

% parameter seed gen

s = 1.01+rand() * 0.2;
m = (density) *(0.2+ rand()*(0.05)); % mean of gaussian, controls density of rain
v = intensity + rand(1,1)*0.3; % variance of gaussian,  controls intensity of rain streak
l = randi(40)+20; % len of motion blur, control size of rain streak

% Generate proper noise seed

dense_chnl = zeros(h,w, 1);
dense_chnl_noise = imnoise(dense_chnl, 'gaussian', m, v);
dense_chnl_noise = imresize(dense_chnl_noise, s, 'bicubic'); 
posv = randi(size(dense_chnl_noise, 1) - h); 
posh = randi(size(dense_chnl_noise, 2) - w);
dense_chnl_noise = dense_chnl_noise(posv:posv+h-1, posh:posh+w-1);

% form filter
filter = fspecial('motion', l, theta);
dense_chnl_motion = imfilter(dense_chnl_noise, filter);

% Generate streak with motion blur
dense_chnl_motion(dense_chnl_motion<0) = 0;
dense_streak = repmat(dense_chnl_motion, [1,1,3]); 

% Render Rain streak
tr = rand()*0.05+ 0.04*l + 0.2;
image_rain = image_rain + tr * dense_streak; % render dense rain image
image_rain(image_rain >= 1) = 1;
actual_streak = image_rain - img; 
end
