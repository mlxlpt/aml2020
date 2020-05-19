%% A mettre au propre :
%paramsM.kernel_width = 1.0;
%paramsM.noise_level = [0.01,0.1,0.5,1.0];
paramsM.noise_level = 0.05;
paramsM.noiseCV = false; % ne pas changer, en test
paramsM.kernel_width = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25];
paramsM.kernel_type = 'squaredexponential';
paramsM.useLogScale = false;

F_fold    = 10;     % # of Folds for cv
valid_ratio  = 0.5;    % train/test ratio

% plot inside
metrics = cross_validation_gpr( X', Y', F_fold, valid_ratio, paramsM , true); 