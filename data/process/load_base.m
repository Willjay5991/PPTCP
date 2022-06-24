function [ feas, labs ] = load_base( path )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% output:
% 			feas: n*d
% 			labs: 1*n
% by yjiedu@foxmail.com
dataset = load(path);
feas = double(dataset.resnet50_features);
labs = double(dataset.labels);
end
