%PEJO, ERVIN JOSH
%RAFAEL, EDGAR JR.
%TN36

image = imread('images/mouse_512.jpg');
gray_image = rgb2hsv(image); 
binary_image = imbinarize(gray_image); % Convert to binary

% Remove noise
binary_image = bwareaopen(binary_image, 50); % Remove small objects

% Connected Component Analysis
CC = bwconncomp(binary_image); % Find connected components
stats = regionprops(CC, 'Centroid', 'BoundingBox', 'Area'); % Extract properties

% Display results
imshow(image); hold on;
for i = 1:length(stats)
    rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
    plot(stats(i).Centroid(1), stats(i).Centroid(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
end
title('Object Detection using Connected Components');
hold off;
