function [ Src_name_list,Tar_name_list,path ] = load_data( dataset_name )
%LOAD_DATA Summary of this function goes here
%   Detailed explanation goes here
%   dataset_name: 'ImageCLEF', 'decaf6', 'Office31', 'OfficeHome'
% yjiedu@foxmail.com
root = 'data/';
if strcmp(dataset_name, 'ImageCLEF')
    Src_name_list = 'CCIIPP';
    Tar_name_list = 'IPCPCI';
    path = 'ImageCLEF/';
elseif strcmp(dataset_name, 'Office31')
    Src_name_list = 'AADDWW';
    Tar_name_list = 'DWAWAD';
    path = 'Office31/';
elseif strcmp(dataset_name, 'OfficeHome')
    Src_name_list = 'AAACCCPPPRRR';
    Tar_name_list = 'CPRAPRACRACP';
    path = 'OfficeHome/';
elseif strcmp(dataset_name, 'decaf6')
    Src_name_list = 'AAACCCDDDWWW';
    Tar_name_list = 'CDWADWACWACD';
    path = 'decaf6/';
    
end
path = [root, path];
end

