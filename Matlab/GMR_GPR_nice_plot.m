%% Initialisation
clear all;
close all;
clc;

%% Loading the data
data_to_load = "genfct"; %"shalegas" "drawing" "genfct"

if (strcmp(data_to_load, "shalegas"))
    data = load('../dataset/shalegas_data.csv');
    fid = fopen('../dataset/shalegas_header.csv');
    header = textscan(fid,'%s%s%s%s%s%s%s%s%s%s%s%s%s','Delimiter',',');
    fclose(fid);
    celldisp(header);
    X_index = false(1, numel(header));
    Y_index = false(1, numel(header));
    for i=1:numel(header)
        X_index(i) = strcmp(header{i}{1}, 'Bakken (ND-MT)');% || ...
                       %strcmp(header{i}{1}, 'Barnett (TX)');
        Y_index(i) = strcmp(header{i}{1}, 'Permian (TX-NM)') ;%|| ...
                       %strcmp(header{i}{1}, 'Barnett (TX)');
    end
    X = data(:, X_index);
    Y = data(:, Y_index);
elseif (strcmp(data_to_load, "drawing")) 
    limits = [-50 50 -50 50];
    data = ml_generate_mouse_data(limits, 'labels');
    close;
    
    X = data(1,:)';
    Y = data(2,:)';
elseif (strcmp(data_to_load, "genfct"))
    data =load('genfct.csv');
    X = data(:,1);
    Y = data(:,2);
end

%normalising the data
Y = Y - mean(Y);

validSize = 0.0;
[X_train, Y_train, X_test, Y_test] = split_data(X, Y, validSize);

if (size(X, 2)+size(Y, 2)<=2)
    x_grid = [min(X)-1:0.1:max(X)+1];
    figure();
    scatter(X_train, Y_train, 'g');
    hold on;
    if(~isempty(X_test))
        scatter(X_test, Y_test, 'r');
    end
elseif (size(X, 2)+size(Y, 2)<=3)
    figure();
    scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'g');
    hold on;
    if(~isempty(X_test))
        scatter3(X_test(:, 1), X_test(:, 2), Y_test, 'r');
    end
end



%% Selecting the method :
method = "GPR"; %"GMR","GPR"

%% GPR
if strcmp(method, "GPR")
    %parameters :
    noise_level = 0.1;
    kernel_width = 1;
    kernel_type = 'squaredexponential';

    gprMdl = fitrgp(X_train, Y_train,'FitMethod', 'none', 'BasisFunction','none',...
    'Sigma', noise_level, 'ConstantSigma', true, 'KernelFunction', kernel_type, ...
    'KernelParameters', [kernel_width; 1], 'OptimizeHyperparameters', 'none');
    
    if (size(X, 2)+size(Y, 2)<=2)
        [y_pred,~,y_int] = predict(gprMdl,x_grid');
        figure();
        plot(X_train,Y_train,'g.', 'MarkerSize', 25);
        hold on
        if(~isempty(X_test))
            plot(X_test,Y_test,'r.', 'MarkerSize', 25);
        end
        plot(x_grid,y_pred,'b');
        plot(x_grid, y_int(:,2), '--k')
        plot(x_grid, y_int(:,1), '--k')
        xlabel('x', 'FontSize',18);
        ylabel('y', 'FontSize',18);
        if(~isempty(X_test))
            legend('train data', 'test data','Fit', '95% confidence interval',  'FontSize',18);
        else
            legend('train data', 'Fit', '95% confidence interval',  'FontSize',18);
        end
        title(['GPR fit with a RBF kernel with sigma : ', num2str(noise_level),...
               ' and kernel width : ', num2str(kernel_width)], 'FontSize',24);
        hold off
    end
    GPR_loss = loss(gprMdl,X,Y);
end

%% GMR
if strcmp(method, "GMR")
    Xi = [X_train, Y_train]';
    params.cov_type = 'full';
    params.k = 7;
    params.max_iter_init = 100;
    params.max_iter = 500;
    params.d_type = 'L2';
    params.init = 'plus';

    % Run GMM-EM function, estimates the paramaters by maximizing loglik
    [Priors, Mu, Sigma] = gmmEM(Xi, params);
    
    
    N = size(X,2); P = size(Y,2);
    in  = 1:N;
    out = N+1:(N+P);
    
    if (size(X, 2)+size(Y, 2)<=2)
        plot_gmm(Xi, Priors, Mu, Sigma, params, 'Final Estimates for EM-GMM');
        [y_pred, var_est] = gmr(Priors, Mu, Sigma, x_grid, in, out);
        var_est = squeeze(var_est);
        figure();
        plot(X_train,Y_train,'g.', 'MarkerSize', 25);
        hold on
        if(~isempty(X_test))
            plot(X_test,Y_test,'r.', 'MarkerSize', 25);
        end
        plot(x_grid, y_pred,'k', 'LineWidth', 2);
        plot(x_grid, y_pred+var_est', '--k', 'LineWidth', 3)
        plot(x_grid, y_pred-var_est', '--k', 'LineWidth', 3)
        xlabel('x', 'FontSize',18);
        ylabel('y', 'FontSize',18);
        if(~isempty(X_test))
            l = legend('train data', 'test data','Fit', 'var estimates', 'FontSize',18);
        else
            l = legend('train data','Fit', 'var estimates', 'FontSize',18);
        end
        
        title(['GMR fit with ', params.cov_type, ' covariance : ', 'and ' num2str(params.k),...
               ' Gaussian components'], 'FontSize',24);
        hold off
    end
    [y_pred, var_est] = gmr(Priors, Mu, Sigma, X', in, out);
    GMR_loss = 1/length(y_pred)*norm(y_pred-Y, 2);

end
%% 