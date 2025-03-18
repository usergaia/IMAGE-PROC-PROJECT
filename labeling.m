clc; clear; close all;

% Define folders correctly
burjFolderPath = 'dataset/burj/';
bballFolderPath = 'dataset/basketball/';
carFolderPath = 'dataset/car/';
fishFolderPath = 'dataset/clownfish/';
mouseFolderPath = 'dataset/mouse/';

% Get list of images
burjFiles = dir(fullfile(burjFolderPath, '*.jpg'));
bballFiles = dir(fullfile(bballFolderPath, '*.jpg'));
carFiles = dir(fullfile(carFolderPath, '*.jpg'));
fishFiles = dir(fullfile(fishFolderPath, '*.jpg'));
mouseFiles = dir(fullfile(mouseFolderPath, '*.jpg'));

% Initialize arrays to store feature data
features = [];
names = {};
labels = [];


% Process images correctly
for k = 1:length(burjFiles)
    imgPath = fullfile(burjFolderPath, burjFiles(k).name); % Use original folder path
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = burjFiles(k).name;
        labels = [labels; 0];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', burjFiles(k).name);
    end
end

% Process class images (basketball, label = 1)
for k = 1:length(bballFiles)
    imgPath = fullfile(bballFolderPath, bballFiles(k).name);
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    % Ensure the extracted feature vector has the correct size
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = bballFiles(k).name;
        labels = [labels; 1];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', bballFiles(k).name);
    end
end



% Process class images (car, label = 2)
for k = 1:length(carFiles)
    imgPath = fullfile(carFolderPath, carFiles(k).name);
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    % Ensure the extracted feature vector has the correct size
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = carFiles(k).name;
        labels = [labels; 2];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', carFiles(k).name);
    end
end

% Process class images (car, label = 3)
for k = 1:length(fishFiles)
    imgPath = fullfile(fishFolderPath, fishFiles(k).name);
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    % Ensure the extracted feature vector has the correct size
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = fishFiles(k).name;
        labels = [labels; 3];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', fishFiles(k).name);
    end
end

% Process class images (clown fish, label = 4)
for k = 1:length(mouseFiles)
    imgPath = fullfile(mouseFolderPath, mouseFiles(k).name);
    img = imread(imgPath);
    featureVector = extractFeatures(img);
    
    % Ensure the extracted feature vector has the correct size
    if numel(featureVector) == 28
        features = [features; featureVector];
        names{end+1} = mouseFiles(k).name;
        labels = [labels; 4];
    else
        fprintf('Skipping %s due to incorrect feature extraction.\n', mouseFiles(k).name);
    end
end

% Ensure features matrix is not empty before saving
if isempty(features)
    error('No valid features extracted. Check image formats and GLCM computation.');
end

% Save features, labels, and image names into a .mat file
save('feature_extracted.mat', 'features', 'labels', 'names');

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
        glcm = glcm + glcm'; % Make symmetric
        glcm = glcm / sum(glcm(:)); % Normalize
        
        % Extract texture features
        [I, J] = meshgrid(1:size(glcm, 2), 1:size(glcm, 1));
        I = I(:);
        J = J(:);
        
        energy = sum(glcm(:).^2);
        contrast = sum(glcm(:) .* (I - J).^2);
        homogeneity = sum(glcm(:) ./ (1 + (I - J).^2));
        entropy = -sum(glcm(glcm > 0) .* log(glcm(glcm > 0))); 
       
        
        % Extract color histogram features (normalized)
        if size(img, 3) == 3
            rHist = imhist(img(:,:,1), 8) / numel(img(:,:,1)); 
            gHist = imhist(img(:,:,2), 8) / numel(img(:,:,2)); 
            bHist = imhist(img(:,:,3), 8) / numel(img(:,:,3)); 
        else
            rHist = zeros(8,1);
            gHist = zeros(8,1);
            bHist = zeros(8,1);
        end

        % Combine all features into one vector
        featureVector = [energy, contrast, homogeneity, entropy, rHist', gHist', bHist'];

    catch ME
        fprintf('Error processing image: %s\n', ME.message);
        featureVector = NaN(1, 28);
    end
end
