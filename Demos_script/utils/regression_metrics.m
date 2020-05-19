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