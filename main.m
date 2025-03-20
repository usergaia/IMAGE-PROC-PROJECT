%% PEJO, ERVIN JOSH
%% RAFAEL, EDGAR JR.
%% TN36

clc; clear; close all;

%% Step 1: Read and Resize Image

% uncomment the desired image to be processed

img = imread('images/raw/ball.jpg'); % input image
exp_ext_obj = imread('images/ground_truth/ball.jpg'); % expected extracted object (ground truth)
gt_mask = imread('images/ground_truth_mask/ball_m.jpg'); % ground truth mask

% img = imread('images/raw/bk.jpg'); % input image
% exp_ext_obj = imread('images/ground_truth/bk.jpg'); % expected extracted object (ground truth)
% gt_mask = imread('images/ground_truth_mask/bk_m.jpg'); % ground truth mask

%img = imread('images/raw/car.jpg'); % input image
%exp_ext_obj = imread('images/ground_truth/car.jpg'); % expected extracted object (ground truth)
%gt_mask = imread('images/ground_truth_mask/car_m.jpg'); % ground truth mask

%img = imread('images/raw/cf.jpg'); % input image
%exp_ext_obj = imread('images/ground_truth/cf.jpg'); % expected extracted object (ground truth)
%gt_mask = imread('images/ground_truth_mask/cf_m.jpg'); % ground truth mask

%img = imread('images/raw/lm.jpg'); % input image
%exp_ext_obj = imread('images/ground_truth/lm.jpg'); % expected extracted object (ground truth)
%gt_mask = imread('images/ground_truth_mask/lm_m.jpg'); % ground truth mask

img = imresize(img, [512 512]); 
exp_ext_obj = imresize(exp_ext_obj, [512 512]);
gt_mask = imresize(gt_mask, [512 512]); 

%% Step 2: Apply Canny Edge Detection and Morphological Gradient

% first get grayscale image for edge detection
gray_img = rgb2gray(img);

% apply canny edge detection
edges = edge(gray_img, 'Canny', [0.04, 0.15]); 

% apply morphological gradient for edge enhancement
se = strel('disk', 2);
morph_gradient = imsubtract(imdilate(edges, se), imerode(edges, se));

%% Step 3: Create Initial Mask and Reduce Noise with Morphological Operations
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

%% Step 4: Apply K-means Only on the Masked Region with the Color Space LAB
% extract pixels within the initial mask for clustering
[rows, cols, ~] = size(img);
lab_img = rgb2lab(img); % convert to lab for better color-based segmentation
reshaped_lab = reshape(lab_img, rows * cols, 3);
masked_indices = initial_mask(:);
masked_pixels = reshaped_lab(masked_indices, :); % reshape into 2D array for k-means

% apply k-means clustering on the masked pixels
num_clusters = 2;
[cluster_idx, ~] = kmeans(masked_pixels, num_clusters, 'Replicates', 5);

% create a clustered image for the masked region
clustered_mask = zeros(rows * cols, 1);
clustered_mask(masked_indices) = cluster_idx; 
clustered_image = reshape(clustered_mask, rows, cols);

% determine which cluster corresponds to the object (typically the larger one)
cluster_counts = histcounts(cluster_idx, 1:num_clusters+1);
[~, main_cluster] = max(cluster_counts);
kmeans_mask = (clustered_image == main_cluster) & initial_mask;

%% Step 5: Refine K-means Segmentation with Morphological Operations

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
    final_mask = initial_mask; 
end

%% Step 6: Extract Objects Using Final Mask
ext_foreground_object = zeros(size(img), 'uint8');
for c = 1:3
    ext_foreground_object(:,:,c) = img(:,:,c) .* uint8(final_mask); %extract foreground for every pixel that is in the mask
end

ext_background_object = zeros(size(img), 'uint8');
for c = 1:3
    ext_background_object(:,:,c) = img(:,:,c) .* uint8(~final_mask); %extract background for every pixel that is not in the mask
end

%% EXTRA : Modify the Extracted Objects (Gaussian Blur Background and Comic Style Foreground)

%%% FOREGROUND %%%

% get the dimensions of the input image
[h, w, ~] = size(img);

% define the number of posterization levels for cartoon effect
posterization_levels = 8;  

% apply posterization to the entire foreground
posterized_img = floor(double(ext_foreground_object) / (256 / posterization_levels)) * (256 / posterization_levels);  
posterized_img = uint8(posterized_img);

% boost the colors throughout the entire foreground to enhance vibrancy
color_boost_factor = 5;  
boosted_colors = posterized_img * color_boost_factor;  
boosted_colors(boosted_colors > 255) = 255;  
comic_foreground = uint8(boosted_colors);  

% apply saturation boost to make colors more vibrant throughout
hsv_img = rgb2hsv(comic_foreground);
hsv_img(:,:,2) = hsv_img(:,:,2) * 1.4; % Increase saturation
hsv_img(:,:,2) = min(hsv_img(:,:,2), 1); % Cap at maximum saturation
comic_foreground = hsv2rgb(hsv_img);
comic_foreground = uint8(comic_foreground * 255);

% add overall warm tone to the entire foreground
warm_factor = [1.1, 1.0, 0.85]; 
for c = 1:3
    temp = double(comic_foreground(:,:,c)) * warm_factor(c);
    temp(temp > 255) = 255;
    comic_foreground(:,:,c) = uint8(temp);
end

% create edges for outline effect
gray_foreground = rgb2gray(ext_foreground_object);
comic_edges = edge(gray_foreground, 'Canny', [0.1, 0.2]);
comic_edges = bwmorph(comic_edges, 'thin');
comic_edges = comic_edges & final_mask;

% add black outlines
line_thickness = 1;
thick_edges = imdilate(comic_edges, strel('disk', line_thickness));
for c = 1:3
    comic_foreground(:,:,c) = comic_foreground(:,:,c) .* uint8(~thick_edges);
end

% apply the comic style only where the foreground mask exists
comic_foreground = comic_foreground .* uint8(repmat(final_mask, [1 1 3]));

%%% BACKGROUND %%%
% apply gaussian blur to the background
sigma = 10; 
blurred_background = imgaussfilt(ext_background_object, sigma);

%%% COMBINE %%%

comic_result = blurred_background; % initially pass the blurred background to the final image
for c = 1:3
    comic_result(:,:,c) = comic_result(:,:,c) + comic_foreground(:,:,c) .* uint8(final_mask); % add the comic foreground on top of the blurred background
end

%% Step 7: Display Results
% create a full screen figure
figure('Units', 'normalized', 'Position', [0 0 1 1], ...
       'Name', 'Segmentation Results', 'NumberTitle', 'off');


subplot(5,4,[1,2,3,4]), imshow(img), title('Original Image', 'FontSize', 14, 'FontWeight', 'bold');

subplot(5,4,5), imshow(edges), title('1. Canny Edge Detection', 'FontSize', 12);
subplot(5,4,6), imshow(morph_gradient), title('2. Morphological Gradient', 'FontSize', 12);
subplot(5,4,7), imshow(initial_mask), title('3. Initial Masking', 'FontSize', 12);
subplot(5,4,8), imshow(clustered_image, []), title('4. K-means Clustering', 'FontSize', 12);

subplot(5,4,9), imshow(kmeans_mask), title('5. K-means Segmentation', 'FontSize', 12);
subplot(5,4,10), imshow(final_mask), title('6. Refined Segmentation', 'FontSize', 12);
subplot(5,4,11), imshow(ext_foreground_object), title('7. Extracted Foreground', 'FontSize', 12);
subplot(5,4,12), imshow(ext_background_object), title('8. Extracted Background', 'FontSize', 12);

subplot(5,4,14), imshow(blurred_background), title('Gaussian Blurred Background', 'FontSize', 12);
subplot(5,4,15), imshow(comic_foreground), title('Comic Style Foreground + Golden Glow', 'FontSize', 12);

subplot(5,4,[17,18,19,20]), imshow(comic_result), title('Final Output', 'FontSize', 14, 'FontWeight', 'bold');

%% Step 8: Object Detection on Segmented Object in the Image

% Convert extracted foreground object to binary for region analysis
extracted_binary = rgb2gray(ext_foreground_object) > 0;

% Extract region properties from the final mask
stats = regionprops(final_mask, 'Centroid', 'BoundingBox');

% Load the pre-trained KNN model
load('models/knnModel.mat', 'knnModel');

% Create figure for displaying results
figure('Name', 'OUTPUT', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(img);
title('BEFORE');

subplot(1, 2, 2);
imshow(comic_result);
title('AFTER');
hold on;

% Iterate through each detected region
for i = 1:length(stats)
    % Get the bounding box coordinates
    bbox = stats(i).BoundingBox;
    
    % Extract features from the RGB foreground object
    obj_features = extractImageFeatures(ext_foreground_object);

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


disp('IMAGE SEGMENTATION AND OBJECT DETECTION COMPLETED!');

%% PERFORMANCE EVALUATION (SEGMENTATION | IoU)
disp('------------------------------------------------------|');
disp('EVALUATING SEGMENTATION PERFORMANCE...');
disp('------------------------------------------------------|');

% preprocess the ground truth mask
if gt_mask == 3
    gray_gt = gt_mask(:, :, 1); % convert to grayscale if it has 3 channels
    binary_gt = imbinarize(gray_gt); 
else
    binary_gt = imbinarize(gt_mask); % convert to binary directly if it is already grayscale
end

if final_mask == 3
    gray_pred = final_mask(:, :, 1); % convert to grayscale if it has 3 channels
    final_mask = imbinarize(gray_pred); 
end

% remove small objects and fill holes in the ground truth mask
binary_gt = bwareaopen(binary_gt, 100); 
binary_gt = imfill(binary_gt, 'holes'); 

% convert ground truth and predicted masks to binary (if they are not already)%
disp('Confirming Binary Masks...');
if any(binary_gt(:) > 1) || any(final_mask(:) > 1)
    binary_gt = binary_gt > 0;  
    final_mask = final_mask > 0;
end

disp('Unique values in binary ground truth mask (Expected Extracted Mask):'); 
disp(unique(binary_gt));

disp('Unique values in binary predicted mask (Actual Extracted Mask):');
disp(unique(final_mask));

% calculate the intersection and union of the ground truth and predicted masks
disp('Calculating Intersection over Union (IoU) Accuracy...');
intersection = sum(binary_gt(:) & final_mask(:));  
union = sum(binary_gt(:) | final_mask(:));  

% calculate IoU (Intersection ÷ Union)
if union == 0
    iou = 0;
else
    iou = (intersection / union) * 100;
end

disp('=================================================|');
fprintf('IoU Accuracy: %.2f%%\n', iou);
disp('=================================================|');

% display both ground truth and predicted masks
figure('Name', 'COMPARISON: Expected VS Actual', 'NumberTitle', 'off');
subplot(2, 2, 1);
imshow(exp_ext_obj);
title('Expected Extracted Object');

subplot(2, 2, 2);
imshow(ext_foreground_object);
title('Actual Extracted Object');

subplot(2, 2, 3);
imshow(binary_gt);
title('Binary Ground Truth Mask');

subplot(2, 2, 4);
imshow(final_mask);
title('Binary Predicted Mask');



