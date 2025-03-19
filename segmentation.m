%% PEJO, ERVIN JOSH
%% RAFAEL, EDGAR JR.
%% TN36

clc; clear; close all;

%% Step 1: Read and Resize Image
img = imread('images/car.jpg');
img = imresize(img, [512 512]); 

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

% get the dimensions of the input image (height, width, and channels) 
[h, w, ~] = size(img);  % ignore channel because we don't need it

% define the number of posterization levels to reduce unique colors for more cartoonish effect.  
posterization_levels = 8;  

% apply posterization by grouping pixel values into fewer shades
posterized_img = floor(double(ext_foreground_object) / (256 / posterization_levels)) * (256 / posterization_levels);  
posterized_img = uint8(posterized_img); % convert back to 8-bit format for proper display.  

% create edges of the extracted object to highlight key boundaries
gray_foreground = rgb2gray(ext_foreground_object);  % convert the foreground image to grayscale since edge detection works best without color information
comic_edges = edge(gray_foreground, 'Canny', [0.1, 0.2]); % detect edges using the canny method to highlight key boundaries
comic_edges = bwmorph(comic_edges, 'thin');  % thin the detected edges to remove excess pixels
comic_edges = comic_edges & final_mask; % retain only the edges within the segmented object to prevent unwanted outlines outside the subject

% boost the colors of the posterized image to enhance vibrancy
color_boost_factor = 1.3;  
boosted_colors = posterized_img * color_boost_factor;  

% clamp color values to a maximum of 255 to prevent overflow and ensure valid pixel intensities
boosted_colors(boosted_colors > 255) = 255;  

% convert the color-boosted image back to 8-bit format for proper display and function compatibility 
comic_foreground = uint8(boosted_colors);  

% thicken the detected edges to strengthen the black outlines
line_thickness = 1;  
thick_edges = imdilate(comic_edges, strel('disk', line_thickness));  

% remove colors where thick edges appear to ensure black outlines remain visible over the colored image
for c = 1:3  
    comic_foreground(:,:,c) = comic_foreground(:,:,c) .* uint8(~thick_edges);  
end  

%%% BACKGROUND %%%
% apply gaussian blur to the background
sigma = 10; 
blurred_background = imgaussfilt(ext_background_object, sigma);

%%% COMBINE %%%

comic_result = blurred_background; % initially pass the blurred background to the final image
for c = 1:3
    comic_result(:,:,c) = comic_result(:,:,c) + comic_foreground(:,:,c) .* uint8(final_mask); % add the comic foreground on top of the blurred background
end

%% Step 9: Display Results
% create a full screen figure
figure('Units', 'normalized', 'Position', [0 0 1 1]);

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
subplot(5,4,15), imshow(comic_foreground), title('Comic Style Foreground', 'FontSize', 12);

subplot(5,4,[17,18,19,20]), imshow(comic_result), title('Final Output', 'FontSize', 14, 'FontWeight', 'bold');

%% Step 10: Object Detection on Segmented Object in the Image

% Convert extracted foreground object to binary for region analysis
extracted_binary = rgb2gray(ext_foreground_object) > 0;

% Extract region properties from the final mask
stats = regionprops(final_mask, 'Centroid', 'BoundingBox');

% Load the pre-trained KNN model
load('knnModel.mat', 'knnModel');

% Create figure for displaying results
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(comic_result);
title('Object Detection and Classification');
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