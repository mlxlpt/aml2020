function [y_est, var_est] = gmr(Priors, Mu, Sigma, X, in, out)
%GMR This function performs Gaussian Mixture Regression (GMR), using the 
% parameters of a Gaussian Mixture Model (GMM) for a D-dimensional dataset,
% for D= N+P, where N is the dimensionality of the inputs and P the 
% dimensionality of the outputs.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o X:       N x M array representing M datapoints of N dimensions.
%   o in:      1 x N array representing the dimensions of the GMM parameters
%                to consider as inputs.
%   o out:     1 x P array representing the dimensions of the GMM parameters
%                to consider as outputs. 
% Outputs ----------------------------------------------------------------
%   o y_est:     P x M array representing the retrieved M datapoints of 
%                P dimensions, i.e. expected means.
%   o var_est:   P x P x M array representing the M expected covariance 
%                matrices retrieved. 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = length(in);
P = length(out);
M = size(X, 2);

MuX = Mu(1:N, :);
MuY = Mu(N+1:end, :);
SigmaXX = Sigma(1:N, 1:N, :);
SigmaYX = Sigma(N+1:end, 1:N, :);
SigmaXY = Sigma(1:N, N+1:end, :);
SigmaYY = Sigma(N+1:end, N+1:end, :);

gmrparams.k = size(Mu,2);
K = gmrparams.k;
beta = expectation_step(X, Priors, MuX, SigmaXX, gmrparams);

MuTilt = zeros(P, M, K);
for k = 1 : K
    MuTilt(:, :, k) = MuY(:, k) + SigmaYX(:, :, k)*inv(SigmaXX(:, :, k))*(X-MuX(:,k));
end

y_est = zeros(P, M);
for m = 1 : M
    for k = 1 : K 
        y_est(:, m) = y_est(:, m) + beta(k, m)*MuTilt(:, m, k);
    end
end

SigmaTilt=zeros(P, P, K);
for k = 1 : K
    SigmaTilt(:, :, k) = SigmaYY(:,:,k)-SigmaYX(:, :, k)*inv(SigmaXX(:,:,k))*SigmaXY(:,:,k);
end

var_est = zeros(P, P, M);
for m = 1 : M
    for k = 1 : K
        var_est(:, :, m) = var_est(:, :, m) + beta(k, m)*(MuTilt(:, m, k)^2+SigmaTilt(:, :, k));
       
    end
    var_est(:,:,m) = var_est(:,:,m) - y_est(:, m)^2;
end



end

