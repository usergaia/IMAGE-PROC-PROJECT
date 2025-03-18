clc; clear; close all;

% Define folders for positive (basketball) and negative (non-basketball) classes
posFolder = 'dataset/basketball/'; % Folder containing basketball images (label = 1)
negFolder = 'dataset/non-basketball/'; % Folder containing non-basketball images (label = 0)

% Get list of images in both folders
posFiles = dir(fullfile(posFolder, '*.jpg'));
negFiles = dir(fullfile(negFolder, '*.jpg'));

% Initialize arrays to store feature data
features = [];
names = {};
labels = [];

% Process positive class images (basketball, label = 1)
for k = 1:length(posFiles)
    imgPath = fullfile(posFolder, posFiles(k).name);
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    % Ensure the extracted feature vector has the correct size
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = posFiles(k).name;
        labels = [labels; 1];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', posFiles(k).name);
    end
end

% Process negative class images (non-basketball, label = 0)
for k = 1:length(negFiles)
    imgPath = fullfile(negFolder, negFiles(k).name);
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    % Ensure the extracted feature vector has the correct size
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = negFiles(k).name;
        labels = [labels; 0];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', negFiles(k).name);
    end
end

% Ensure features matrix is not empty before saving
if isempty(features)
    error('No valid features extracted. Check image formats and GLCM computation.');
end

% Save features, labels, and image names into a .mat file
save('texture_color_features.mat', 'features', 'labels', 'names');

fprintf('Feature extraction completed. Data saved to texture_color_features.mat\n');

% =======================================================================
%                     FEATURE EXTRACTION FUNCTION
% =======================================================================
function featureVector = extractFeatures(img)
    try
        % Convert to grayscale for GLCM feature extraction
        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end
        
        % Compute GLCM features
        glcm = graycomatrix(gray, 'NumLevels', 8, 'Offset', [0 1]); 
        glcm = glcm + glcm';
        glcm = glcm / sum(glcm(:)); % Normalize
        
        % Create index matrices
        [I, J] = meshgrid(1:size(glcm, 2), 1:size(glcm, 1));
        I = I(:);
        J = J(:);
        
        % Extract texture features
        energy = sum(glcm(:).^2);
        contrast = sum(glcm(:) .* (I - J).^2);
        homogeneity = sum(glcm(:) ./ (1 + (I - J).^2));
        entropy = -sum(glcm(glcm > 0) .* log(glcm(glcm > 0))); 
        
        % Extract color histogram features (normalized)
        if size(img, 3) == 3  % Ensure image is in RGB
            rHist = imhist(img(:,:,1), 8) / numel(img(:,:,1)); % Red channel histogram
            gHist = imhist(img(:,:,2), 8) / numel(img(:,:,2)); % Green channel histogram
            bHist = imhist(img(:,:,3), 8) / numel(img(:,:,3)); % Blue channel histogram
        else
            rHist = zeros(8,1);
            gHist = zeros(8,1);
            bHist = zeros(8,1);
        end

        % Combine all features into one vector
        featureVector = [energy, contrast, homogeneity, entropy, rHist', gHist', bHist'];
    catch ME
        fprintf('Error processing image: %s\n', ME.message);
        featureVector = NaN(1, 28); % 4 texture + 24 color features
    end
end
