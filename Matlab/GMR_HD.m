%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
restoredefaultpath();
userpath('clear');


addpath(genpath('../../ML_toolbox'))
T = readtable('../dataset/kin8nm.csv', 'HeaderLines',0); 


disp('ATTENTION, CE SCRIPT A BESOIN DE LA ML TOOLBOX!!!');
disp('Entrée pour lancer le script')
pause


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    2) Learn the GMM Model from your regression data       %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
M=table2array(T);
M=M(1:end,:);
X=M(:,[1:8]); % X: theta i=1..8 joint pos
y=M(:,9); %y is column 9
y_true = y;
Xi = [X  y]'; close all;

%%

% Fit GMM with Chosen parameters
%K = 13;

%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
%[Priors, Mu, Sigma] = ml_gmmEM(Xi, K);
%%

N = size(X,2); P = size(y,2); M = size(X,1);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions


k_range = [5,8];
F_fold = 10;
valid_ratio = 0.5;

[N,M] = size(X);
[P,M] = size(y);
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
        [y_est, ~] = ml_gmr(Priors, Mu, Sigma, X', in, out);
        [sd_folds(4,f), sd_folds(5,f)] = gmm_metrics([X_train; y_train],Priors,Mu,Sigma,'full');

        [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_est, y');
        fprintf('.');
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


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    3) Validate my_gmr.m function on 2D Dataset    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute Regressive signal and variance
N = size(X,2); P = size(y,2); M = size(X,1);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions
[y_est, var_est] = ml_gmr(Priors, Mu, Sigma, X', in, out);

% Function handle for my_gmr.m
f = @(X) ml_gmr(Priors,Mu,Sigma,X, in, out);

% Plotting Options for Regressive Function
options           = [];
options.title     = 'Estimated y=f(x) from Gaussian Mixture Regression';
options.regr_type = 'GMR';
options.surf_type = 'surf';
ml_plot_value_func(X,f,[1 2],options);hold on

% Plot Training Data
options = [];
options.plot_figure = true;
options.points_size = 12;
options.labels = zeros(M,1);
options.plot_labels = {'$x_1$','$x_2$','y'};
ml_plot_data([X y],options);

%%
function [] = plot_gmr_cross_validation(metrics, k_range)
%PLOT_GMR_CROSS_VALIDATION Summary of this function goes here
%   Detailed explanation goes here
%% Plot GMM Model Selection Metrics for F-fold cross-validation with std
figure;

subplot(1,2,1)
errorbar(k_range',metrics.mean_AIC', metrics.std_AIC','--or','LineWidth',2); hold on;
errorbar(k_range',metrics.mean_BIC', metrics.std_BIC','--ob','LineWidth',2);
grid on
xlabel('Number of K components','FontSize',16); ylabel('AIC/BIC Score','FontSize',16)
legend('AIC', 'BIC','FontSize',16)
title('GMM Model Selection Metrics','FontSize',20)

%% Plot Regression Metrics for F-fold cross-validation with std
subplot(1,2,2)
[ax,hline1,hline2]=plotyy(k_range',metrics.mean_MSE',[k_range' k_range'],[metrics.mean_NMSE' metrics.mean_R2']);
delete(hline1);
delete(hline2);
hold(ax(1),'on');
errorbar(ax(1),k_range', metrics.mean_MSE', metrics.std_MSE','--o','LineWidth',2,'Color', [0 0.447 0.741]);
hold(ax(2),'on');
errorbar(ax(2),k_range',metrics.mean_NMSE', metrics.std_NMSE','--or','LineWidth',2);
errorbar(ax(2),k_range',metrics.mean_R2', metrics.std_R2','--og','LineWidth',2);
xlabel('Number of K components','FontSize',16); ylabel('Measures','FontSize',16)
legend('MSE', 'NMSE', '$R^2$','Interpreter','Latex','FontSize',16)
grid on
title('Regression Metrics','FontSize',20)
end



function [ X_train, y_train, X_test, y_test ] = split_regression_data(X, y, valid_ratio )
%SPLIT_DATA Randomly partitions a dataset into train/test sets using
%   according to the given tt_ratio
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y        : (1 x M), a vector with labels y \in {0,1} corresponding to X.
%       o valid_ratio : train/test ratio.
%   output ----------------------------------------------------------------
%
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary Variable
[~, M] = size(X);
tt_ratio = 1-valid_ratio;

% Randomize Dataset Indices
rand_idx  = randperm(M); 

% Compute expected number of Training/Testing Points
M_train    = round(M*tt_ratio);
M_test     = M - M_train;

% Split the dataset in train and test subsets
train_idx = rand_idx(1:M_train);
X_train   = X(:,train_idx);
y_train   = y(train_idx);

test_idx  = rand_idx(M_train+1:end);
X_test   = X(:,test_idx);
y_test   = y(test_idx);

end

function [ logl ] = gmmLogLik(X, Priors, Mu, Sigma)
%MY_GMMLOGLIK Compute the likelihood of a set of parameters for a GMM
%given a dataset X
%
%   input------------------------------------------------------------------
%
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                    Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%
%   output ----------------------------------------------------------------
%
%      o logl       : (1 x 1) , loglikelihood
%%


% Auxiliary Variables
[N, M] = size(X);
[~, K] = size(Mu);

%Compute the likelihood of each datapoint for each K
P_xi = zeros(K,M);
for i=1:K
    P_xi(i,:) = gaussPDF(X, Mu(:,i), Sigma(:,:,i));
end

%Compute the total log likelihood
alpha_P_xi = Priors*P_xi;
alpha_P_xi(alpha_P_xi < realmin) = realmin;
logl = sum(log(alpha_P_xi));

end



function [AIC, BIC] =  gmm_metrics(X, Priors, Mu, Sigma, cov_type)
%GMM_METRICS Computes the metrics (AIC, BIC) for model fitting
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o cov_type : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%
%   output ----------------------------------------------------------------
%
%       o AIC      : (1 x 1), Akaike Information Criterion
%       o BIC      : (1 x 1), Bayesian Information Criteria
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[N, M] = size(X);
[~, K] = size(Mu);

% Compute GMM Likelihood
[ ll ] = gmmLogLik(X, Priors, Mu, Sigma);

% Compute B Parameters
switch cov_type
    case 'full' % (Equation 15)
        B = K * (1 + 2*N + N*(N-1)/2) - 1;
    case 'diag' % (Equation 16)
        B = K * (1 + 2*N) - 1;
    case 'iso'  % (Equation 17)
        B = K * (1 + N + 1) - 1;
end

% Compute AIC (Equation 13)
% AIC = ml_aic(ll, B, 2);
AIC = -2*ll + 2*B;

% Compute AIC (Equation 14)
% BIC = ml_bic(ll, B, M);
BIC = -2*ll + log(M)*B;
end

function [MSE, NMSE, Rsquared] = regression_metrics( yest, y )
%REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o R squared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = size(yest, 2);
mu = mean(y, 2);
mu_hat = mean(yest, 2);

MSE = sum((yest-y).^2)/M;

VAR = sum((y-mu).^2);
NMSE = (M-1)*MSE/VAR;

num=zeros(size(mu));
denom1=num;
denom2=num;
for m=1:M
   num = num + (y(:,m)-mu)*(yest(:,m)-mu_hat);
   denom1 = denom1 + (y(:,m)-mu)^2;
   denom2 = denom2 + (yest(:,m)-mu_hat)^2;
end
Rsquared = num^2/(denom1*denom2);

if isnan(Rsquared)
    Rsquared = 0;
end

end
