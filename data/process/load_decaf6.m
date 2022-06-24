function [ feas, labs ] = load_decaf6( path )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% yjiedu@foxmail.com
dataset = load(path);
feas = double(dataset.feas);
labs = double(dataset.labels);
feas = normr(feas);
labs = (labs - 1)';
end

