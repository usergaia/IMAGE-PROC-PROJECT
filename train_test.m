% Load extracted features and labels
load('basketball_features.mat', 'features', 'labels');

% Split dataset: 80% for training, 20% for testing
cv = cvpartition(length(labels), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

% Separate training and testing data
X_train = features(trainIdx, :);
Y_train = labels(trainIdx, :);
X_test = features(testIdx, :);
Y_test = labels(testIdx, :);

% Train KNN classifier with k=5 (you can tune this later)
knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', 5, 'Standardize', 1);

% Predict labels for test data
Y_pred = predict(knnModel, X_test);

% Compute accuracy
accuracy = sum(Y_pred == Y_test) / length(Y_test) * 100;
fprintf('KNN Classification Accuracy: %.2f%%\n', accuracy);

% Plot confusion matrix
figure;
confusionchart(Y_test, Y_pred);
title('Confusion Matrix for KNN Classification');

% Save the trained KNN model
save('knnModel.mat', 'knnModel');
