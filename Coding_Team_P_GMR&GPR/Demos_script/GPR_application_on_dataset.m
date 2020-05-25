%% GPR for kin8nm dataset
clc; clear all; close all;
restoredefaultpath();
userpath('clear');
addpath(genpath('utils')); % For split regression data

%% Loading the dataset :
T = readtable('./datasets/kin8nm.csv', 'HeaderLines',0); 
Mdata=table2array(T);
X=Mdata(:,[1:8]); % X: theta i=1..8 joint pos
y=Mdata(:,9); %length from robot base is column 9
y=y-mean(y); %substracting the mean of the data
y_true = y;
Xi = [X  y]';
[N,M] = size(X);
[P,M] = size(y);

%% run cross validation GPR
paramsM.noise_level = [0.0075, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5];
paramsM.kernel_width = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5];
paramsM.kernel_type = 'squaredexponential';
paramsM.useLogScale = false; %Sometime we evaluate kernel chosen with log law

optimizeParameterWithMatlab = false; % USE MATLAB OPTIMIZER?
paramsM.plotMSEcolorMap = false; 
paramsM.plotR2Var = true;

F_fold    = 10;     % # of Folds for cv
valid_ratio  = 0.5;    % train/test ratio

optimMaxIter = F_fold * length(paramsM.noise_level) * length(paramsM.kernel_width);
if (optimMaxIter > 200)
    optimMaxIter = 200;
end
%This is to do as many optimizer restart as the grid search computations

if optimizeParameterWithMatlab == false
    % Plots are inside this function
    metrics = cross_validation_gpr( X', y', F_fold, valid_ratio, paramsM); 
else    
    [X_train, Y_train, X_test, y_test] = split_regression_data(X',y',valid_ratio);
    
    gprMdl = fitrgp(X_train', Y_train','FitMethod', 'none','BasisFunction','none','KernelFunction','squaredexponential',...
    'OptimizeHyperparameters',{'KernelScale', 'Sigma'},'KernelParameters', [1, 1],'HyperparameterOptimizationOptions',...
    struct('MaxObjectiveEvaluations',optimMaxIter));
    %struct('AcquisitionFunctionName','expected-improvement-plus'));

    [y_pred,~,~] = predict(gprMdl,X_test');
    [mse, nsme, r2] = regression_metrics(y_pred', y_test);
    fprintf('R2: %.3f - MSE: %.3f - NMSE: %.3f\n', r2, mse, nsme);
end

function [metrics] = cross_validation_gpr( X, y, F_fold, valid_ratio, params)
    [N,M] = size(X);
    [P,M] = size(y);
    metrics.mean_MSE = zeros(length(params.noise_level), length(params.kernel_width));
    metrics.mean_NMSE = metrics.mean_MSE;
    metrics.mean_R2 = metrics.mean_MSE;
    metrics.std_MSE = metrics.mean_MSE;
    metrics.std_NMSE = metrics.mean_MSE;
    metrics.std_R2 = metrics.mean_MSE;

    sd_folds = zeros(4, F_fold);

    kernel_type = params.kernel_type;
    noise_level = params.noise_level;
    paramPair = [];
    r2vect = [];
    msevect = [];
    r2var = [];
    
    e=1;
    k_range = params.kernel_width;
    
    % loop over noise
    for nse = params.noise_level
        fprintf('\n\nNoise %.4f', nse);
        k=1;
        % loop over widths
        for kernel_wdth=k_range
            fprintf('\nCV kwidth = %.4f: ', kernel_wdth);
            paramPair = [paramPair;[nse,kernel_wdth]];
            %F-folds
            for f=1:F_fold
                [X_train, Y_train, X_test, y_test] = split_regression_data(X,y,valid_ratio);

                gprMdl = fitrgp(X_train', Y_train','FitMethod', 'none', 'BasisFunction','none',...
                'Sigma', nse, 'ConstantSigma', true, 'KernelFunction', kernel_type, ...
                'KernelParameters', [kernel_wdth; 1], 'OptimizeHyperparameters', 'none');
                [y_pred,~,~] = predict(gprMdl,X_test');
                [sd_folds(1,f), sd_folds(2,f), sd_folds(3,f)] = regression_metrics(y_pred', y_test);
                fprintf('.');
            end

            metrics.mean_MSE(e,k) = mean(sd_folds(1,:));
            metrics.mean_NMSE(e,k) = mean(sd_folds(2,:));
            metrics.mean_R2(e,k) = mean(sd_folds(3,:));
            metrics.std_MSE(e,k) = std(sd_folds(1,:));
            metrics.std_NMSE(e,k) = std(sd_folds(2,:));
            metrics.std_R2(e,k) = std(sd_folds(3,:));
            r2vect = [r2vect;metrics.mean_R2(e,k)];
            msevect = [msevect;metrics.mean_MSE(e,k)];
            r2var = [r2var;metrics.std_R2(e,k)];
            k = k + 1;
        end
        e = e + 1;
    end

    if length(params.noise_level) == 1
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
        % Color map in case of 2D CV (len(noise) > 1)
        %Prepare grid (noise & kernel width may not be sorted)
        kmin = min(params.kernel_width);
        kmax = max(params.kernel_width);
        kgrid = [kmin:0.001:kmax]';
        emin = min(params.noise_level);
        emax = max(params.noise_level);
        egrid = [emin:0.001:emax]';
        
        %build meshgrid, interpolate and make colormap
        F = scatteredInterpolant(paramPair,r2vect);
        [xGrid, yGrid] = meshgrid(egrid, kgrid);
        xq = xGrid(:);
        yq = yGrid(:);
        vq = F(xq, yq);
        fittedImage = reshape(vq,  length(kgrid),length(egrid));
        c = hot;
        figure;
        imagesc(fittedImage, 'XData',kgrid,'YData',egrid);
        hold on;
        title('$R^2$ evolution with $l$ and $\sigma_e$', 'Interpreter', 'latex', 'FontSize',20);
        xlabel('Kernel width $l$ ', 'FontSize',18, 'Interpreter', 'latex');
        ylabel('Noise $\sigma_e$', 'FontSize',18, 'Interpreter', 'latex');
        hold on;
        colormap(c);
        
        colorbar;
        hold off;
        if params.plotMSEcolorMap == true
            F = scatteredInterpolant(paramPair,msevect);
            vq = F(xq, yq);
            fittedImage = reshape(vq,  length(kgrid),length(egrid));
            c = hot;
            figure;
            imagesc(fittedImage, 'XData',kgrid,'YData',egrid);
            hold on;
            title('MSE evolution with $l$ and $\sigma_e$', 'Interpreter', 'latex', 'FontSize',20);
            xlabel('Kernel width $l$ ', 'FontSize',18, 'Interpreter', 'latex');
            ylabel('Noise $\sigma_e$', 'FontSize',18, 'Interpreter', 'latex');
            colormap(c);
            colorbar;
            hold off;
        end
        if params.plotR2Var == true
            F = scatteredInterpolant(paramPair,r2var);
            vq = F(xq, yq);
            fittedImage = reshape(vq,  length(kgrid),length(egrid));
            c = hot;
            figure;
            imagesc(fittedImage, 'XData',kgrid,'YData',egrid);
            hold on;
            title('std($R^2$) with $l$ and $\sigma_e$', 'Interpreter', 'latex', 'FontSize',20);
            xlabel('Kernel width $l$ ', 'FontSize',18, 'Interpreter', 'latex');
            ylabel('Noise $\sigma_e$', 'FontSize',18, 'Interpreter', 'latex');
            colormap(c);
            colorbar;
            hold off;
        end
    end
end
