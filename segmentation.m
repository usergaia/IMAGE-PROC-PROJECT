%% PEJO, ERVIN JOSH
%% RAFAEL, EDGAR JR.
%% TN36

clc; clear; close all;

%% Step 1: Read and Resize Image
img = imread('images/burj.jpg');
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
initial_mask = imclose(initial_mask, strel('disk', 5)); % connect nearby edges
initial_mask = imopen(initial_mask, strel('disk', 2));  % remove small noise
initial_mask = bwareaopen(initial_mask, 500);           % remove small disconnected regions

% keep only the largest object if there are multiple components
CC_initial = bwconncomp(initial_mask);
if CC_initial.NumObjects > 0
    numPixels = cellfun(@numel, CC_initial.PixelIdxList);
    [~, idx_largest] = max(numPixels);
    initial_mask = false(size(initial_mask));
    initial_mask(CC_initial.PixelIdxList{idx_largest}) = true;
end

%% Step 4: Apply K-means only on the masked region with the color space LAB
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
final_mask = imclose(final_mask, strel('disk', 5)); % connect nearby parts
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
ext_foreground_object = zeros(size(img), 'uint8');
for c = 1:3
    ext_foreground_object(:,:,c) = img(:,:,c) .* uint8(final_mask);
end

ext_background_object = zeros(size(img), 'uint8');
for c = 1:3
    ext_background_object(:,:,c) = img(:,:,c) .* uint8(~final_mask);
end

%% Step 7: Create Gradient Background (decorative purpose)

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
result_with_new_bg = uint8(new_bg*255) + ext_foreground_object;

%% Step 9: Display Results

figure('Position', [100, 100, 1200, 800]);

subplot(3,3,1), imshow(img), title('Original Image');
subplot(3,3,2), imshow(edges), title('Canny Edge Detection');
subplot(3,3,3), imshow(initial_mask), title('Initial Masking (Edge-Detected)');
subplot(3,3,4), imshow(clustered_image, []), title('K-means Clustering Result');
subplot(3,3,5), imshow(kmeans_mask), title('Initial K-means Segmentation');
subplot(3,3,6), imshow(final_mask), title('Refined K-means Segmentation');
subplot(3,3,7), imshow(ext_background_object), title('Extracted Background Object');
subplot(3,3,8), imshow(ext_foreground_object), title('Extracted Foreground Object');
subplot(3,3,9), imshow(result_with_new_bg), title('Foreground with New Background');


%% Step 10: Object Detection on Segmented Object in the Image

% Convert extracted foreground object to binary for region analysis
extracted_binary = rgb2gray(ext_foreground_object) > 0;

% Extract region properties from the final mask
stats = regionprops(final_mask, 'Centroid', 'BoundingBox');

% Load the pre-trained KNN model
load('knnModel.mat', 'knnModel');
disp(size(knnModel.X));

% Create figure for displaying results
figure;
imshow(img);
title('Object Detection and Classification');
hold on;

% Iterate through each detected region
for i = 1:length(stats)
    % Get the bounding box coordinates
    bbox = stats(i).BoundingBox;
    
    % Extract features from the RGB foreground object
    obj_features = extractImageFeatures(ext_foreground_object);
    disp(size(obj_features));
    
    % Reshape features to match the KNN model input format
    obj_features = reshape(obj_features, 1, []);
    
    % Predict the object class using the KNN model
    predictedLabel = predict(knnModel, obj_features);
    
    % Assign appropriate label text based on the prediction
    if predictedLabel == 0
        labelText = 'Burj Khalifa';
    elseif predictedLabel == 1
        labelText = 'Basketball';
    elseif predictedLabel == 2
        labelText = 'Car';
    elseif predictedLabel == 3
        labelText = 'Clown Fish';
    elseif predictedLabel == 4
        labelText = 'Logitech Mouse';
    end
    
    % Draw the bounding box around the detected object
    rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
    
    % Display the class label
    text(bbox(1), bbox(2)-10, labelText, 'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold');
end
hold off;

%% Feature Extraction Function
function featureVector = extractImageFeatures(img)
    try
        % Standardize image size to ensure consistent feature extraction
        img = imresize(img, [256 256]);
        
        % Convert to grayscale for texture analysis if RGB
        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end
        
        % Compute Gray Level Co-occurrence Matrix (GLCM) for texture features
        glcm = graycomatrix(gray, 'NumLevels', 8, 'Offset', [0 1]);
        glcm = glcm + glcm';  % Make symmetric
        glcm = glcm / sum(glcm(:));  % Normalize
        
        % Set up the coordinate system for GLCM calculations
        [I, J] = meshgrid(1:size(glcm, 2), 1:size(glcm, 1));
        I = I(:);
        J = J(:);
        
        % Calculate Haralick texture features from the Gray Level Co-occurrence Matrix (GLCM)
        energy = sum(glcm(:).^2);
        contrast = sum(glcm(:) .* (I - J).^2);
        homogeneity = sum(glcm(:) ./ (1 + (I - J).^2));
        entropy = -sum(glcm(glcm > 0) .* log(glcm(glcm > 0)));
        
        % Compute color histogram features with normalization
        if size(img, 3) == 3
            % Extract and normalize color channel histograms
            rHist = imhist(img(:,:,1), 8) / numel(img(:,:,1));
            gHist = imhist(img(:,:,2), 8) / numel(img(:,:,2));
            bHist = imhist(img(:,:,3), 8) / numel(img(:,:,3));
        else
            % Handle grayscale images by using empty color histograms
            rHist = zeros(8,1);
            gHist = zeros(8,1);
            bHist = zeros(8,1);
        end
        
        % Combine texture and color features into a single feature vector
        featureVector = [energy, contrast, homogeneity, entropy, rHist', gHist', bHist'];
    catch ME
        % Handle any errors during feature extraction
        fprintf('Error processing image: %s\n', ME.message);
        featureVector = NaN(1, 28);  % Return NaN vector with proper dimensions
    end
end
