% demo start
% by yjiedu@foxmail.com

clc;
clear;
% clear res/ dir
delete('res/*.mat');
disp('have clear the res dir');
nowstr = datestr(now,'HH-MM-SS_dd-mm-yyyy');
func_name = 'P2TCP_api';

% add path and load hyperparam
path1 = './P2TCP/param';
path2 = './P2TCP';
addpath(path1)
addpath(path2);
load_param;  % load datasets, funcs, param_opt

diary(['log/' nowstr '_log.txt']); % print output into txt
diary on; % start 

for i=1:length(datasets)
%     disp(param_opt);
    param_opt.dataset = datasets{i};
    param_opt.func = funcs{i};
    param_opt.i = i;
    param_opt.file_name = [num2str(i),'th'];
    main(param_opt);
    disp('##############################################');
    disp('##############################################');
end

diary off; % stop recorder
% rmpath
rmpath(path1);
rmpath(path2);
