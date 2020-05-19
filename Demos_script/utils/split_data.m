function [X_train, Y_train, X_test, Y_test] = split_data(X, Y, validSize)
%trainTestSplit Split a dataset between train and test data

% Calculate the number of elements
nbSamples = size(X, 1);

idx = randperm(nbSamples);
X = X(idx, :);
Y = Y(idx, :);

nbTrainSamples = floor(nbSamples * (1 - validSize));
nbTestSamples = nbSamples - nbTrainSamples;

% Initialize all returned arrays to empty vectors
X_train = X(1:nbTrainSamples, :);
Y_train = Y(1:nbTrainSamples, :);
X_test = X(nbTrainSamples+1:end, :);
Y_test = Y(nbTrainSamples+1:end, :);
end


