% Define dataset folders
basketballPath = 'dataset/basketball/';
nonBasketballPath = 'dataset/non-basketball/';

% Get list of images
basketballImages = dir(fullfile(basketballPath, '*.jpg')); 
nonBasketballImages = dir(fullfile(nonBasketballPath, '*.jpg')); 

% Initialize feature and label arrays
features = [];
labels = [];

% Process basketball images (label = 1)
for i = 1:length(basketballImages)
    img = imread(fullfile(basketballPath, basketballImages(i).name));  
    featureVector = extractFeatures(img);
    features = [features; featureVector];
    labels = [labels; 1];  % Basketball label
end

% Process non-basketball images (label = 0)
for i = 1:length(nonBasketballImages)
    img = imread(fullfile(nonBasketballPath, nonBasketballImages(i).name));  
    featureVector = extractFeatures(img);
    features = [features; featureVector];
    labels = [labels; 0];  % Non-basketball label
end

% Normalize entire feature matrix (row-wise normalization)
features = normalize(features, 'range');

% Save extracted features and labels
save('basketball_features.mat', 'features', 'labels');

% ---- FEATURE EXTRACTION FUNCTION ----
function featureVector = extractFeatures(img)
    % Resize image for consistency
    img = imresize(img, [256 256]);  

    % Ensure image is RGB
    if size(img, 3) == 1  
        % Convert grayscale to RGB (repeat channels)
        img = cat(3, img, img, img);
    end

    % Convert to grayscale for texture extraction
    grayImg = rgb2gray(img);

    %% 🔹 Color Histogram Extraction (Fixed Length)
    numBins = 32; % Ensure consistent bin size
    histRed = imhist(img(:,:,1), numBins) / numel(img(:,:,1)); 
    histGreen = imhist(img(:,:,2), numBins) / numel(img(:,:,2)); 
    histBlue = imhist(img(:,:,3), numBins) / numel(img(:,:,3)); 
    colorFeatures = [histRed' histGreen' histBlue'];  % (1 × 96)

    %% 🔹 Texture Feature Extraction (GLCM - Fixed Size)
    offsets = [0 1; -1 1; -1 0; -1 -1]; 
    glcm = graycomatrix(grayImg, 'Offset', offsets, 'NumLevels', 32); 
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

    % Ensure GLCM features have a fixed length (take mean)
    textureFeatures = [mean(stats.Contrast), mean(stats.Correlation), ...
                       mean(stats.Energy), mean(stats.Homogeneity)]; % (1 × 4)

    %% 🔹 Final Feature Vector (Fixed Size)
    featureVector = [colorFeatures, textureFeatures];  % (1 × 100)
end

