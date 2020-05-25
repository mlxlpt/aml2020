%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
restoredefaultpath();
userpath('clear');
addpath(genpath('gmmbox')); addpath('plot_functions'); addpath('utils');

disp('This script uses functions from the ML toolbox developped at LASA EPFL');
disp('Press enter to start the script')
pause

%% Defining the parameters for the crossvalidation :
k_range = [2:10];
F_fold = 10;
valid_ratio = 0.5;

%% Loading the dataset :
T = readtable('datasets/kin8nm.csv', 'HeaderLines',0); 
M=table2array(T);
M=M(1:end,:);
X=M(:,[1:8]); % X: theta i=1..8 joint pos
y=M(:,9); %y is column 9
y_true = y;
Xi = [X  y]'; close all;
N = size(X,2); P = size(y,2); M = size(X,1);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions
[N,M] = size(X);
[P,M] = size(y);

%%Crossvalidation :
%Structure to store the metrics
metrics.mean_MSE = zeros(1, length(k_range));
metrics.mean_NMSE = metrics.mean_MSE;
metrics.mean_R2 = metrics.mean_MSE;
metrics.mean_AIC = metrics.mean_MSE;
metrics.mean_BIC = metrics.mean_MSE;
metrics.std_MSE = metrics.mean_MSE;
metrics.std_NMSE = metrics.mean_MSE;
metrics.std_R2 = metrics.mean_MSE;
metrics.std_AIC = metrics.mean_MSE;
metrics.std_BIC = metrics.mean_MSE;

sd_folds = zeros(5, F_fold);

i=1;
for k=k_range
    fprintf('\nCV k= %d: ', k);

    for f=1:F_fold
        [X_train, y_train, X_test, y_test] = split_regression_data(X',y',valid_ratio);

        [Priors, Mu, Sigma] = ml_gmmEM([X_train ;  y_train], k);
        [y_est, ~] = ml_gmr(Priors, Mu, Sigma, X_test, in, out);
        [sd_folds(4,f), sd_folds(5,f)] = gmm_metrics([X_train; y_train],Priors,Mu,Sigma,'full');

        [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_est, y_test);
    end
    metrics.mean_MSE(i) = mean(sd_folds(1,:));
    metrics.mean_NMSE(i) = mean(sd_folds(2,:));
    metrics.mean_R2(i) = mean(sd_folds(3,:));
    metrics.mean_AIC(i) = mean(sd_folds(4,:));
    metrics.mean_BIC(i) = mean(sd_folds(5,:));
    metrics.std_MSE(i) = std(sd_folds(1,:));
    metrics.std_NMSE(i) = std(sd_folds(2,:));
    metrics.std_R2(i) = std(sd_folds(3,:));
    metrics.std_AIC(i) = std(sd_folds(4,:));
    metrics.std_BIC(i) = std(sd_folds(5,:));
    i=i+1;
end
fprintf('\n');
plot_gmr_cross_validation(metrics, k_range)
