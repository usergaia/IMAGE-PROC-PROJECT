clc;
clear;
close all;

% Read the image
img = imread('t2.jpg');

% convert to grayscale (assuming the image is not already in grayscale)
gimg = rgb2gray(img);  

% get pixels that are  greater than or equal to 1 (binarize)
bimg = gimg >= 1;  

% refine
bimg = bwareaopen(bimg, 100); 
bimg = imfill(bimg, 'holes');  

figure; % compare the original image and the mask
subplot(1, 2, 1);
imshow(img);
subplot(1, 2, 2);
imshow(bimg);


% imwrite(bimg, 'test2.jpg'); % uncomment to save the mask


