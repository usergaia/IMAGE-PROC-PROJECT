%% This script is separate from the main file as it handles additional modifications to each K-means cluster and combines them to extract the two fish objects.

clc; clear; close all;

%% Step 1: Read and Resize Image
% - Standardize dimensions with 512x512 to be consistent with the results
img = imread('images/fish.jpg');  
img = imresize(img, [512 512]); 

%% Step 2: Convert to Lab Color Space and Apply Histogram Equalization
% - Equalizing only L channel improves contrast without affecting colors
% - This makes fish features stand out while preserving natural coloration
lab_img = rgb2lab(img); 
lab_img(:,:,1) = histeq(lab_img(:,:,1)); 
img_eq = lab2rgb(lab_img);

%% Step 3: Convert to AB Color Space for K-Means
% - Focus only on color components (a,b), ignoring luminance differences 
% - This helps cluster pixels based on color similarity regardless of lighting
ab = double(lab_img(:,:,2:3)); % Extracts (row, column, channels); 2:3 selects the a and b color channels
ab = reshape(ab, [], 2); % Reshapes for K-means clustering, which requires 2D data

%% Step 4: Apply K-means Clustering to Segment Image by Color Similarity
% - k=3 means the image was processed into three clusters
% - [kmeans] passes cluster labels to idx and cluster centroids to centers.
k = 3; 
[idx, centers] = kmeans(ab, k, 'Replicates', 3);

%% Step 5: Ensure Consistent Cluster Order by Sorting Based on Brightness
L_values = zeros(k, 1);
L_channel = lab_img(:,:,1);  
L_flat = L_channel(:);       
for i = 1:k
    L_values(i) = mean(L_flat(idx == i)); 
end

[~, sorted_order] = sort(L_values, 'descend'); 

%% Step 6: Reassign Cluster Labels for Stable Segmentation Results
% - This guarantees that the indices of each cluster is consistent all the time
new_idx = zeros(size(idx));
for i = 1:k
    new_idx(idx == sorted_order(i)) = i; 
end
idx = new_idx; 

%% Step 7: Convert Clustered Data to Image Format
segmented_img = reshape(idx, size(lab_img,1), size(lab_img,2));

%% Step 8: Create Segment Masks for Targeted Processing
% - Each mask isolates a specific color cluster
% - Original colors are preserved for each segment
segment_masks = cell(k, 1);
segment_imgs = cell(k, 1);

for i = 1:k
    segment_masks{i} = (segmented_img == i);
    segment_imgs{i} = zeros(size(img), 'uint8');
    
    for c = 1:3
        segment_imgs{i}(:,:,c) = img(:,:,c) .* uint8(segment_masks{i});
    end
end

%% Step 9: Extract White Parts of the Clownfish from Segment 3
% - White areas need special handling due to their high brightness
% - Low saturation + high value identifies white regions regardless of hue
segment3_hsv = rgb2hsv(segment_imgs{3});
white_mask = segment_masks{3} & (segment3_hsv(:,:,2) < 0.2) & (segment3_hsv(:,:,3) > 0.7);

%% Step 10: Combine Multiple Segments to Create Comprehensive Fish Mask
% - Merges segment 1, segment 2, and white parts from segment 3
combined_mask = segment_masks{1} | segment_masks{2} | white_mask;

%% Step 11: Clean Mask Using Morphological Operations
% - Remove small noise (imopen)
% - Close gaps in the mask (imclose)
% - Fill holes (imfill)
% - Remove small isolated regions (bwareaopen)
refined_mask = combined_mask;
refined_mask = imopen(refined_mask, strel('disk', 2)); 
refined_mask = imclose(refined_mask, strel('disk', 4)); 
refined_mask = imfill(refined_mask, 'holes');
refined_mask = bwareaopen(refined_mask, 100); 

%% Step 12: Apply Refined Mask to Extract Fish from Original Image
% - Creates final result with accurate fish segmentation
combined_fish = zeros(size(img), 'uint8');
for c = 1:3
    combined_fish(:,:,c) = img(:,:,c) .* uint8(refined_mask);
end

%% Step 13: Add a Yellow Glow Effect to the Combined Fish Image

% Step 13.1: Create a Glow Mask
% To define the area where the glow will be applied.
glow_mask = imgaussfilt(double(refined_mask), 10); % Apply Gaussian blur to the mask (sigma=10 for a soft glow)

% Step 13.2: Normalize the Glow Mask
% To ensure the glow intensity is within a reasonable range.
glow_mask = glow_mask / max(glow_mask(:)); % Normalize to [0, 1]

% Step 13.3: Apply the yellow glow effect
glow_color = [7, 7, 0]; % Yellow glow (r,g,b)
glow_intensity = 0.5; % Adjust the intensity of the glow (0 to 1)

% Create a glow layer by multiplying the glow mask with the glow color
glow_layer = zeros(size(img), 'double');
for c = 1:3
    glow_layer(:,:,c) = glow_mask * glow_color(c) * glow_intensity;
end

% Step 13.4 Combine the glow layer with the original fish image
glowing_fish = double(combined_fish) / 255; % Convert to double for blending
glowing_fish = glowing_fish + glow_layer; % Add the glow effect
glowing_fish = glowing_fish / max(glowing_fish(:)); % Normalize to avoid oversaturation
glowing_fish = uint8(glowing_fish * 255); % Convert back to uint8 for display

%% Step 14: Add Background to the Detected Fish

% Step 14.1: Create a Background Mask
% isolates the background from the original image.
background_mask = ~refined_mask; % Invert the refined mask to get the background

% Step 14.2: Extract the Background
% get the original background without the fish.
background = zeros(size(img), 'uint8');
for c = 1:3
    background(:,:,c) = img(:,:,c) .* uint8(background_mask);
end

% Step 14.3: Combine the Fish and Background
% place the detected fish (glowing fish) back into the original image.
final_glowing_image_with_background = background + glowing_fish; % For the glowing fish

% Step 14.4: Display the Final Images with Background
figure('Position', [100, 100, 1200, 450], 'Name', 'Final Images with Background');
subplot(1, 2, 1), imshow(img), title('Original Image');
subplot(1, 2, 2), imshow(final_glowing_image_with_background), title('Glowing Fish with Background');

%% Step 15: Object Detection
% todo



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DOCUMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - Uncomment these to show each cluster segmentations

%% Display each cluster segmentation
%figure('Position', [100, 100, 900, 450], 'Name', 'Fish Segment Combination');
%subplot(2, 3, 1), imshow(img), title('Original Image');
%subplot(2, 3, 5), imshow(white_mask), title('White Parts of the clownfish');
%subplot(2, 3, 6), imshow(combined_fish), title('Combined Fish');
%for i = 1:k
%    subplot(2, 3, i+1), imshow(segment_imgs{i}), title(['Segment ' num2str(i)]);
%end

