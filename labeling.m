clc; clear; close all;

% Define dataset folders
burjFolderPath = 'dataset/burj-clean/';
bballFolderPath = 'dataset/basketball-clean/';
carFolderPath = 'dataset/car-clean/';
fishFolderPath = 'dataset/clownfish-clean/';
mouseFolderPath = 'dataset/mouse-clean/';

% Load image files from each folder
burjFiles = dir(fullfile(burjFolderPath, '*.jpg'));
bballFiles = dir(fullfile(bballFolderPath, '*.jpg'));
carFiles = dir(fullfile(carFolderPath, '*.jpg'));
fishFiles = dir(fullfile(fishFolderPath, '*.jpg'));
mouseFiles = dir(fullfile(mouseFolderPath, '*.jpg'));

% Initialize storage for features, image names, and labels
features = [];
names = {};
labels = [];

% Process images and extract features
processImages(burjFiles, burjFolderPath, 0);
processImages(bballFiles, bballFolderPath, 1);
processImages(carFiles, carFolderPath, 2);
processImages(fishFiles, fishFolderPath, 3);
processImages(mouseFiles, mouseFolderPath, 4);

% Save extracted feature data if valid
if isempty(features)
    error('No valid features extracted. Check image formats and GLCM computation.');
end

save('feature_extracted.mat', 'features', 'labels', 'names');
fprintf('Feature extraction completed. Data saved to feature_extracted.mat\n');

% Function to process images and extract features
function processImages(imageFiles, folderPath, label)
    for k = 1:length(imageFiles)
        imgPath = fullfile(folderPath, imageFiles(k).name);
        img = imread(imgPath);
        featureVector = extractFeatures(img);
        
        if numel(featureVector) == 28
            features = [features; featureVector];
            names{end+1} = imageFiles(k).name;
            labels = [labels; label];
        else
            fprintf('Skipping %s due to incorrect feature extraction.\n', imageFiles(k).name);
        end
    end
end

% Function to extract features from an image
function featureVector = extractFeatures(img)
    try
        % Convert to grayscale if image is RGB
        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end
        
        % Compute GLCM and normalize
        glcm = graycomatrix(gray, 'NumLevels', 8, 'Offset', [0 1]);
        glcm = glcm + glcm';
        glcm = glcm / sum(glcm(:));
        
        % Extract texture features
        [I, J] = meshgrid(1:size(glcm, 2), 1:size(glcm, 1));
        I = I(:);
        J = J(:);
        
        energy = sum(glcm(:).^2);
        contrast = sum(glcm(:) .* (I - J).^2);
        homogeneity = sum(glcm(:) ./ (1 + (I - J).^2));
        entropy = -sum(glcm(glcm > 0) .* log(glcm(glcm > 0))); 
        
        % Compute color histogram features
        if size(img, 3) == 3
            rHist = imhist(img(:,:,1), 8) / numel(img(:,:,1)); 
            gHist = imhist(img(:,:,2), 8) / numel(img(:,:,2)); 
            bHist = imhist(img(:,:,3), 8) / numel(img(:,:,3)); 
        else
            rHist = zeros(8,1);
            gHist = zeros(8,1);
            bHist = zeros(8,1);
        end
        
        % Combine features into a single vector
        featureVector = [energy, contrast, homogeneity, entropy, rHist', gHist', bHist'];
    catch ME
        fprintf('Error processing image: %s\n', ME.message);
        featureVector = NaN(1, 28);
    end
end
