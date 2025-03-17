%% PEJO, ERVIN JOSH
%% RAFAEL, EDGAR JR.
%% TN36

clc; clear; close all;

%% Step 1: Read and Resize Image
img = imread('images/pyr.jpg');
img = imresize(img, [512 512]); 

%% Step 2: Apply Canny Edge Detection and Morphological Gradient

% first get grayscale image for edge detection
gray_img = rgb2gray(img);

% apply canny edge detection with optimized thresholds
edges = edge(gray_img, 'Canny', [0.04, 0.15]); % adjust threshold as needed

% apply morphological gradient for edge enhancement
se = strel('disk', 2);
morph_gradient = imsubtract(imdilate(edges, se), imerode(edges, se));

%% Step 3: Create Initial Mask and Remove Noise 
% create initial mask by filling holes in the edges
initial_mask = imfill(morph_gradient, 'holes');

% perform basic morphological operations to refine the mask
initial_mask = imclose(initial_mask, strel('disk', 4)); % connect nearby edges
initial_mask = imopen(initial_mask, strel('disk', 2));  % remove small noise
initial_mask = imfill(initial_mask, 'holes');           % fill any remaining holes
initial_mask = bwareaopen(initial_mask, 500);           % remove small disconnected regions

% keep only the largest object if there are multiple components
CC_initial = bwconncomp(initial_mask);
if CC_initial.NumObjects > 0
    numPixels = cellfun(@numel, CC_initial.PixelIdxList);
    [~, idx_largest] = max(numPixels);
    initial_mask = false(size(initial_mask));
    initial_mask(CC_initial.PixelIdxList{idx_largest}) = true;
end


%% Step 4: Apply K-means only on the masked region
% extract pixels within the initial mask for clustering
[rows, cols, ~] = size(img);
lab_img = rgb2lab(img); % convert to lab for better color-based segmentation
reshaped_lab = reshape(lab_img, rows * cols, 3);

% use logical indexing directly instead of find() for better performance
masked_indices = initial_mask(:);
masked_pixels = reshaped_lab(masked_indices, :);

% apply k-means clustering on the masked pixels
num_clusters = 2; % object vs details within the object
[cluster_idx, ~] = kmeans(masked_pixels, num_clusters, 'Replicates', 5);

% create a clustered image for the masked region - use logical indexing
clustered_mask = zeros(rows * cols, 1);
clustered_mask(masked_indices) = cluster_idx; % improved performance with logical indexing
clustered_image = reshape(clustered_mask, rows, cols);

% determine which cluster corresponds to the object (typically the larger one)
cluster_counts = histcounts(cluster_idx, 1:num_clusters+1);
[~, main_cluster] = max(cluster_counts);
kmeans_mask = (clustered_image == main_cluster) & initial_mask;

%% Step 5: Combine Initial and K-means masks for Final Refinement
% use the k-means result to refine the initial mask
final_mask = kmeans_mask;
final_mask = imclose(final_mask, strel('disk', 3)); % connect nearby parts
final_mask = imfill(final_mask, 'holes'); % fill any remaining holes
final_mask = bwareaopen(final_mask, 500); % remove small objects

% keep only the largest connected component
CC_final = bwconncomp(final_mask);
if CC_final.NumObjects > 0
    numPixels = cellfun(@numel, CC_final.PixelIdxList);
    [~, idx_largest] = max(numPixels);
    final_mask = false(size(final_mask));
    final_mask(CC_final.PixelIdxList{idx_largest}) = true;
else
    final_mask = initial_mask; % fallback to initial mask if k-means fails
end

%% Step 6: Extract Object Using Final Mask
extracted_object = zeros(size(img), 'uint8');
for c = 1:3
    extracted_object(:,:,c) = img(:,:,c) .* uint8(final_mask);
end

%% Step 7: Create Simple Gradient Background (placeholder)

% gradient map based on image dimensions
[rows, cols, ~] = size(img);
[y, x] = meshgrid(1:cols, 1:rows);

%  gradient map on each pixel based on its position in the image
gradient_map = (x/rows + y/cols) / 2;
gradient_map = imgaussfilt(gradient_map, 3); 
new_background = zeros(size(img));

% gradient value
new_background(:,:,1) = 0.5 + 0.1 * gradient_map; % Red component
new_background(:,:,2) = 0.5 - 0.3 * gradient_map; % Green component
new_background(:,:,3) = 0.8 - 0.1 * gradient_map; % Blue component

% combine object with new background
background_mask = ~final_mask;
new_bg = zeros(size(img));
for c = 1:3
    new_bg(:,:,c) = new_background(:,:,c) .* double(background_mask);
end
result_with_new_bg = uint8(new_bg*255) + extracted_object;



%% Step 8 : Apply Canny Edge Detection and Morphological Gradient to Simple Gradient Background

% first get grayscale image for edge detection
gray_img2 = rgb2gray(extracted_object);

% apply canny edge detection with optimized thresholds
edges2 = edge(gray_img2, 'Canny', [0.04, 0.15]); % adjust threshold as needed

% apply morphological gradient for edge enhancement
morph_gradient = imsubtract(imdilate(edges2, se), imerode(edges2, se));

%% Step 3: Create Initial Mask and Remove Noise 
% create initial mask by filling holes in the edges
extbg_mask = imfill(morph_gradient, 'holes');

% perform basic morphological operations to refine the mask
extbg_mask = imclose(extbg_mask, strel('disk', 4)); % connect nearby edges
extbg_mask = imopen(extbg_mask, strel('disk', 2));  % remove small noise
extbg_mask = imfill(extbg_mask, 'holes');           % fill any remaining holes
extbg_mask = bwareaopen(extbg_mask, 500);           % remove small disconnected regions


%% Step 9: Display Results
figure('Position', [100, 100, 1200, 800]);

subplot(4, 2, 1), imshow(img), title('Original Image');
subplot(4, 2, 2), imshow(initial_mask), title('Initial Mask');
subplot(4, 2, 3), imshow(kmeans_mask), title('K-means Mask');
subplot(4, 2, 4), imshow(extracted_object), title('Extracted Object');
subplot(4, 2, 5), imshow(result_with_new_bg), title('Object with New Background');
subplot(4, 2, 6), imshow(extbg_mask), title('Applied masking to image w/ new background');

%% Step 10: Object Detection on Segmented Object in the Image

% Extract properties of connected components in the binary mask
stats = regionprops(extbg_mask, 'Centroid', 'BoundingBox', 'Area'); 

% Display the original image
figure, imshow(img); hold on; % Show image and enable overlay drawing

% Loop through each detected object and annotate it
for i = 1:length(stats)
    % Draw a bounding box around the detected object
    rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
    
    % Mark the centroid of the object with a green circle
    plot(stats(i).Centroid(1), stats(i).Centroid(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
end

% Add title to the figure
title('Object Detection using Connected Components');

% Release hold to finalize drawing
hold off;




