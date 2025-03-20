clc;
clear;
close all;

% Read the image
img = imread('images/ground_truth/cf.jpg');

% convert to grayscale (assuming the image is not already in grayscale)
gimg = rgb2gray(img);  

% get pixels that are  greater than or equal to 1 (binarize)
bimg = gimg >= 1;  

% refine
bimg = bwareaopen(bimg, 100); 
bimg = imfill(bimg, 'holes');  

figure;
subplot(1, 2, 1);
imshow(img);
subplot(1, 2, 2);
imshow(bimg);


% imwrite(bimg, 'images/ground_truth_mask/cf_m.jpg'); % uncomment to save the mask


