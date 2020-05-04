function [metrics] = cross_validation_gmr( X, y, F_fold, valid_ratio, k_range, params )
%CROSS_VALIDATION_GMR Implementation of F-fold cross-validation for regression algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (P x M) array representing the y vector assigned to
%                           each datapoints
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o valid_ratio  : (double), Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%       o params    : parameter strcuture of the GMM
%
%   output ----------------------------------------------------------------
%       o metrics : (structure) contains the following elements:
%           - mean_MSE   : (1 x K), Mean Squared Error computed for each value of k averaged over the number of folds.
%           - mean_NMSE  : (1 x K), Normalized Mean Squared Error computed for each value of k averaged over the number of folds.
%           - mean_R2    : (1 x K), Coefficient of Determination computed for each value of k averaged over the number of folds.
%           - mean_AIC   : (1 x K), Mean AIC Scores computed for each value of k averaged over the number of folds.
%           - mean_BIC   : (1 x K), Mean BIC Scores computed for each value of k averaged over the number of folds.
%           - std_MSE    : (1 x K), Standard Deviation of Mean Squared Error computed for each value of k.
%           - std_NMSE   : (1 x K), Standard Deviation of Normalized Mean Squared Error computed for each value of k.
%           - std_R2     : (1 x K), Standard Deviation of Coefficient of Determination computed for each value of k averaged over the number of folds.
%           - std_AIC    : (1 x K), Standard Deviation of AIC Scores computed for each value of k averaged over the number of folds.
%           - std_BIC    : (1 x K), Standard Deviation of BIC Scores computed for each value of k averaged over the number of folds.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

in = 1:N;
out = (N+1):(N+P);

i=1;
for k=k_range
    fprintf('\nCV k= %d: ', k);
    params.k=k;

    for f=1:F_fold
        [X_train, y_train, X_test, y_test] = split_regression_data(X,y,valid_ratio);
        [Priors, Mu, Sigma, ~] = gmmEM([X_train ;  y_train], params);
        [y_est, ~] = gmr(Priors, Mu, Sigma, X_test, in, out);
        [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_est, y_test);
        [sd_folds(4,f), sd_folds(5,f)] = gmm_metrics([X_train; y_train],Priors,Mu,Sigma,params.cov_type);
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
end

