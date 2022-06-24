function [path_opt] = addpath4P2TCP()
%ADDPATH_LOCAL Summary of this function goes here
%   add needed path
% by yjiedu@foxmail.com
fullpath = mfilename('fullpath');
[root,~] = fileparts(fullpath);

path_opt = struct;
path_opt.predict = fullfile(root,'predict'); % classifier path
path_opt.SelfEntropy = fullfile(root,'SelfEntropy');
path_opt.LP = fullfile(root,'LP');


names = fieldnames(path_opt);
for i= 1:length(names)
   addpath(path_opt.(names{i})); 
end

end

