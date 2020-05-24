function [metrics] = cross_validation_gpr( X, y, F_fold, valid_ratio, params, plotLogLik)
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
metrics.mean_MSE = zeros(length(params.noise_level), length(params.kernel_width));
metrics.mean_NMSE = metrics.mean_MSE;
metrics.mean_R2 = metrics.mean_MSE;
metrics.std_MSE = metrics.mean_MSE;
metrics.std_NMSE = metrics.mean_MSE;
metrics.std_R2 = metrics.mean_MSE;
metrics.loglik = metrics.mean_MSE;

sd_folds = zeros(4, F_fold);

kernel_type = params.kernel_type;
noise_level = params.noise_level;
vect = [];
r2vect = [];
e=1;
k_range = params.kernel_width;
for nse = params.noise_level
    fprintf('\n\nnoise %.3f', nse);
    k=1;
    for kernel_wdth=k_range
        fprintf('\nCV kwidth = %.3f: ', kernel_wdth);
        vect = [vect;[nse,kernel_wdth]];
        for f=1:F_fold
            [X_train, Y_train, X_test, y_test] = split_regression_data(X,y,valid_ratio);

            gprMdl = fitrgp(X_train', Y_train','FitMethod', 'none', 'BasisFunction','none',...
            'Sigma', nse, 'ConstantSigma', true, 'KernelFunction', kernel_type, ...
            'KernelParameters', [kernel_wdth; 1], 'OptimizeHyperparameters', 'none');
            [y_pred,~,~] = predict(gprMdl,X_test');
            [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_pred', y_test);
            fprintf('.');
        end
        kfcn = gprMdl.Impl.Kernel.makeKernelAsFunctionOfXNXM(gprMdl.Impl.ThetaHat);
        K_sig = kfcn(X_train',X_train') + eye(size(X_train,2));
        sd_folds(4,1) = 0.5*(Y_train*inv(K_sig)*Y_train'+log(det(K_sig)));

        metrics.mean_MSE(e,k) = mean(sd_folds(1,:));
        metrics.mean_NMSE(e,k) = mean(sd_folds(2,:));
        metrics.mean_R2(e,k) = mean(sd_folds(3,:));
        metrics.std_MSE(e,k) = std(sd_folds(1,:));
        metrics.std_NMSE(e,k) = std(sd_folds(2,:));
        metrics.std_R2(e,k) = std(sd_folds(3,:));
        metrics.loglik(e,k) = sd_folds(4,1);
        r2vect = [r2vect;metrics.mean_R2(e,k)];
        k = k + 1;
    end
    e=e+1;
end

if length(params.noise_level) == 1
    fprintf('\n');
    f = figure();
    if plotLogLik
        subplot(1,2,1)
        %errorbar(params.kernel_width',metrics.loglik', metrics.loglikstd','--or','LineWidth',2); hold on;
        plot(params.kernel_width',metrics.loglik','--or','LineWidth',2); hold on;

        %errorbar(params.kernel_width',metrics.mean_BIC', metrics.std_BIC','--ob','LineWidth',2);
        grid on
        xlabel('Kernel width');% ylabel('Marginal loglik')
        legend('Marginal loglik', 'FontSize',16);
        title('Marginal likelihood','FontSize',20);
        if(params.useLogScale)
            set(gca, 'XScale','log')
            set(gca,'FontSize',16)
        end
        subplot(1,2,2)
    end
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
        set(ax,'FontSize',16)
        xlabel('Kernerl width, log scale');
    else
        xlabel('Kernel width');
    end
    ylabel('Measures','FontSize',16);
    legend('MSE', 'NMSE', '$R^2$','Interpreter','latex','FontSize',18);
    grid on;
    title('GPR Regression Metrics','FontSize',20);
else
    kmin = min(params.kernel_width);
    kmax = max(params.kernel_width);
    kgrid = [kmin:0.01:kmax]';

    emin = min(params.noise_level);
    emax = max(params.noise_level);
    egrid = [emin:0.01:emax]';
    F = scatteredInterpolant(vect,r2vect);

    [xGrid, yGrid] = meshgrid(egrid, kgrid);
    xq = xGrid(:);
    yq = yGrid(:);
    vq = F(xq, yq);
    fittedImage = reshape(vq,  length(kgrid),length(egrid));
    figure;
    imagesc(fittedImage, 'XData',kgrid,'YData',egrid);
    title('$R^2$ evolution with $l$ and $\sigma_e$', 'Interpreter', 'latex', 'FontSize',20);
    xlabel('Kernel width $l$ ', 'FontSize',18, 'Interpreter', 'latex');
    ylabel('Noise $\sigma_e$', 'FontSize',18, 'Interpreter', 'latex');
    hold on;
    colormap;
    colorbar;
    hold off;
end
end