function [metrics] = cross_validation_gpr( X, y, F_fold, valid_ratio, params )
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
metrics.mean_MSE = zeros(1, length(params.kernel_width));
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

kernel_type = params.kernel_type;
noise_level = params.noise_level;

k=1;
if params.noiseCV == false
    k_range = params.kernel_width;

    for kernel_wdth=k_range
        fprintf('\nCV kwidth = %.3f: ', kernel_wdth);

        for f=1:F_fold
            [X_train, Y_train, X_test, y_test] = split_regression_data(X,y,valid_ratio);

            gprMdl = fitrgp(X_train', Y_train','FitMethod', 'none', 'BasisFunction','none',...
            'Sigma', noise_level, 'ConstantSigma', true, 'KernelFunction', kernel_type, ...
            'KernelParameters', [kernel_wdth; 1], 'OptimizeHyperparameters', 'none');
            [y_pred,~,~] = predict(gprMdl,X_test');
            [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_pred', y_test);
            fprintf('.');
        end
        metrics.mean_MSE(k) = mean(sd_folds(1,:));
        metrics.mean_NMSE(k) = mean(sd_folds(2,:));
        metrics.mean_R2(k) = mean(sd_folds(3,:));
        metrics.mean_AIC(k) = NaN; % on met NaN comme ça le plot AIC BIC reste vide (n'existe pas pour GMR/GPR)
        metrics.mean_BIC(k) = NaN;
        metrics.std_MSE(k) = std(sd_folds(1,:));
        metrics.std_NMSE(k) = std(sd_folds(2,:));
        metrics.std_R2(k) = std(sd_folds(3,:));
        metrics.std_AIC(k) = NaN;
        metrics.std_BIC(k) = NaN;

        k = k + 1;
    end
    fprintf('\n');
    f = figure();
    [ax,hline1,hline2]=plotyy(params.kernel_width',metrics.mean_MSE',...
        [params.kernel_width' params.kernel_width'],[metrics.mean_NMSE' metrics.mean_R2']);
    delete(hline1);
    delete(hline2);
    hold(ax(1),'on');
    errorbar(ax(1),params.kernel_width', metrics.mean_MSE', metrics.std_MSE','--o','LineWidth',2,'Color', [0 0.447 0.741]);
    hold(ax(2),'on');
    errorbar(ax(2),params.kernel_width',metrics.mean_NMSE', metrics.std_NMSE','--or','LineWidth',2);
    errorbar(ax(2),params.kernel_width',metrics.mean_R2', metrics.std_R2','--og','LineWidth',2);
    if(params.useLogScale)
        set(ax,'XScale','log');
        xlabel('Kernerl width, log scale');
    else
        xlabel('Kernerl width');
    end
    ylabel('Measures');
    legend('MSE', 'NMSE', '$R^2$','Interpreter','latex','FontSize',14);
    grid on;
    title('GPR Regression Metrics');
else
    if length(params.kernel_width) > 1
        error("Too many kernel width given, expected one but got more");
        return;
    end
    kernel_wdth = params.kernel_width;
    for noise=params.noise_level
        fprintf('\nCV noise = %.4f: ', noise);

        for f=1:F_fold
            [X_train, Y_train, X_test, y_test] = split_regression_data(X,y,valid_ratio);

            gprMdl = fitrgp(X_train', Y_train','FitMethod', 'none', 'BasisFunction','none',...
            'Sigma', noise, 'ConstantSigma', true, 'KernelFunction', kernel_type, ...
            'KernelParameters', [kernel_wdth; 1], 'OptimizeHyperparameters', 'none');
            [y_pred,~,~] = predict(gprMdl,X_test');
            [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_pred', y_test);
            fprintf('.');
        end
        metrics.mean_MSE(k) = mean(sd_folds(1,:));
        metrics.mean_NMSE(k) = mean(sd_folds(2,:));
        metrics.mean_R2(k) = mean(sd_folds(3,:));
        metrics.mean_AIC(k) = NaN; % on met NaN comme ça le plot AIC BIC reste vide (n'existe pas pour GMR/GPR)
        metrics.mean_BIC(k) = NaN;
        metrics.std_MSE(k) = std(sd_folds(1,:));
        metrics.std_NMSE(k) = std(sd_folds(2,:));
        metrics.std_R2(k) = std(sd_folds(3,:));
        metrics.std_AIC(k) = NaN;
        metrics.std_BIC(k) = NaN;

        k = k + 1;
    end
    fprintf('\n');
    f = figure();
    [ax,hline1,hline2]=plotyy(params.noise_level',metrics.mean_MSE',...
        [params.noise_level' params.noise_level'],[metrics.mean_NMSE' metrics.mean_R2']);
    delete(hline1);
    delete(hline2);
    hold(ax(1),'on');
    errorbar(ax(1),params.noise_level', metrics.mean_MSE', metrics.std_MSE','--o','LineWidth',2,'Color', [0 0.447 0.741]);
    hold(ax(2),'on');
    errorbar(ax(2),params.noise_level',metrics.mean_NMSE', metrics.std_NMSE','--or','LineWidth',2);
    errorbar(ax(2),params.noise_level',metrics.mean_R2', metrics.std_R2','--og','LineWidth',2);
    if(params.useLogScale)
        set(ax,'XScale','log');
        xlabel('Noise, log scale');
    else
        xlabel('Noise');
    end
    ylabel('Measures');
    legend('MSE', 'NMSE', '$R^2$','Interpreter','latex','FontSize',19);
    grid on;
    title('GPR Regression Metrics');
end

end