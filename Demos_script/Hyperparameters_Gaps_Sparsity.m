%% Initialisation :
restoredefaultpath();
addpath('utils'); addpath('GMM_GMR_toy_data'); addpath('plot_functions');
clear all; close all; clc;

%% Selecting the data to load :
%genfct : toy dataset
%drawing : to draw a 2D dataset
data_to_load = "genfct"; 

%set to true to test for sparsity or the effect of a gap :
genSparseData = false;
createGapData = false;

%% Selecting the method and its hyperparameters :
method = "GPR"; %"GMR","GPR"

GPR_param.noise_level = 0.1;
GPR_param.kernel_width = 1; 
GPR_param.kernel_type = 'squaredexponential'; %used as RBF 
GPR_param.beta = ones(2, 1)*1; %*1 if one want linear mean fct

GMR_param.cov_type = 'full'; 
GMR_param.k = 9;

%% Selecting the parameters for the plot :
margin = 6; %left and right margin to plot outside the rand of the dataset
  
%% Loading the selected dataset :
if (strcmp(data_to_load, "drawing"))
    limits = [-50 50 -50 50];
    data = ml_generate_mouse_data(limits, 'labels');
    close;
    X = data(1,:)';
    Y = data(2,:)';
    
elseif (strcmp(data_to_load, "genfct"))
    data =load('datasets/genfct.csv');
    X = data(:,1);
    Y = data(:,2);
end

validSize = 0.2; % test/target ratio (inverse than usual)
[X_train, Y_train, X_test, Y_test] = split_data(X, Y, validSize);

if (createGapData) % make gap in data
    [X_train,I]=sort(X_train);
    Y_train= Y_train(I);
    Y_train((10 > X_train) & (X_train > 0)) = [];
    X_train((10 > X_train) & (X_train > 0)) = [];
end
if (genSparseData) %generate sparsity
    I = rand(size(X_train,1),1);
    X_train(I > 0.07) = [];
    Y_train(I > 0.07) = [];
    J = rand(size(X_test,1),1);
    X_test(J > 0.5) = [];
    Y_test(J > 0.5) = [];
end

%% Displaying the data :
figure();
title('Data','FontSize',20);
x_grid = [min(X)-margin:0.1:max(X)+margin];
scatter(X_train, Y_train, 'black');
hold on;
if(~isempty(X_test))
    scatter(X_test, Y_test, 'r');
    legend('Train data','Test data','FontSize',16);
else
    legend('Train data','FontSize',16);
end

%% Applying the chosen method :
% GPR
if strcmp(method, "GPR")
    gprMdl = fitrgp(X_train, Y_train,'FitMethod', 'none', 'BasisFunction', 'linear',...
    'beta', GPR_param.beta, 'Sigma', GPR_param.noise_level, 'ConstantSigma', true, ...
    'KernelFunction', GPR_param.kernel_type, 'KernelParameters', [GPR_param.kernel_width; 1], ...
    'OptimizeHyperparameters', 'none');
    
   
    [y_pred,~,y_int] = predict(gprMdl,x_grid');
    figure();
    if(~isempty(X_test))
        plot1 = plot(X_test,Y_test,'.','color',[0,0,0]+0.5,'MarkerSize', 18);
        %plot1.Color(1) = 0.75;
        hold on;
    end
    plot(X_train,Y_train,'.r' ,'MarkerSize', 18);
    hold on;
    plot(x_grid,y_pred,'b','Linewidth',2);
    plot(x_grid, y_int(:,2), '--k','Linewidth',2)
    plot(x_grid, y_int(:,1), '--k','Linewidth',2)
    xlabel('x', 'FontSize',18);
    ylabel('y', 'FontSize',18);
    if(~isempty(X_test))
        legend('test data', 'train data','Fit', '95% confidence interval',  'FontSize',16);
    else
        legend('train data', 'Fit', '95% confidence interval',  'FontSize',16);
    end
    title(['GPR - RBF kernel with $\sigma$ = ', num2str(GPR_param.noise_level),...
           ' and $l$ = ', num2str(GPR_param.kernel_width)], 'FontSize',20, 'Interpreter', 'latex');
    hold off
end

% GMR
if strcmp(method, "GMR")
    Xi = [X_train, Y_train]';
    params.cov_type = GMR_param.cov_type;
    params.k = GMR_param.k;
    params.max_iter_init = 100;
    params.max_iter = 500;
    params.d_type = 'L2';
    params.init = 'plus';
    
    N = size(X,2); P = size(Y,2); 
    in  = 1:N;
    out = N+1:(N+P);

    % Run GMM-EM function, estimates the paramaters by maximizing loglik
    [Priors, Mu, Sigma] = gmmEM(Xi, params);
    

    plot_gmm(Xi, Priors, Mu, Sigma, params, 'Final Estimates for EM-GMM');
    [y_pred, var_est] = gmr(Priors, Mu, Sigma, x_grid, in, out);
    var_est = squeeze(var_est);
    figure();
    if(~isempty(X_test))
        plot1 = plot(X_test,Y_test,'.','color', [0, 0, 0]+0.5, 'MarkerSize', 25);
        %plot1.Color(1) = 1.0;
        hold on;
    end
    plot(X_train,Y_train,'.r', 'MarkerSize', 25);
    hold on;
    plot(x_grid, y_pred,'b', 'LineWidth', 2.5);
    plot(x_grid, y_pred+var_est', '--k', 'LineWidth', 2)
    plot(x_grid, y_pred-var_est', '--k', 'LineWidth', 2)
    xlabel('x', 'FontSize',18);
    ylabel('y', 'FontSize',18);
    if(~isempty(X_test))
        l = legend('test data', 'train data','Fit', 'var estimates', 'FontSize',18);
    else
        l = legend('train data','Fit', 'var estimates', 'FontSize',18);
    end

    title(['GMR - ', params.cov_type, ' covariance - ', 'and ' num2str(params.k),...
           ' components'], 'FontSize',20,'Interpreter', 'latex')
    hold off   
end
   
