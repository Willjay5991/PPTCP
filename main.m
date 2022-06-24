function main(param_opt)
% iteration framework
% by yjiedu@foxmail.com

path_data = './data'; % dataset
path_util  = './util'; % common function lib
addpath(genpath(path_data)); % add dataset
addpath(genpath(path_util));

% dataset_name: 'ImageCLEF', 'decaf6', 'Office31', 'OfficeHome'
% load dataset
if  isfield(param_opt,'dataset')
    dataset_name = param_opt.dataset;
else
    dataset_name = 'Office31'; % 'Office31'(default)
end
% load func
if isfield(param_opt,'func')
    func = param_opt.func;
else
    func = 'P2TCP_api';%'KNN_learning'(without adaptation);
end
% load file name
if isfield(param_opt, 'file_name')
    file_name = param_opt.file_name;
else
    file_name = 'default';
end


disp(['dataset: ' dataset_name])
[Src_name_list,Tar_name_list,path] = load_data(dataset_name);
acc_matrix =  zeros(1,length(Src_name_list)/2);
time_matrix = zeros(1,length(Src_name_list)/2);

for  i = 1:length(Src_name_list)
        src_name = char(Src_name_list(i));  % source domain name
        tar_name = char(Tar_name_list(i));  % target domain name
        
        disp(['test subtask <<<' src_name,'-' tar_name '>>>'])
        param_opt.stNames = [src_name, tar_name];
		% load src data
        source_data_path = [path, src_name, '.mat'];
        [xs,ys] = feval(['load_',dataset_name],source_data_path); % load feas and label
        ys = ys+1;  
        % load tar data
        target_data_path = [path, tar_name, '.mat'];
        [xt,yt] = feval(['load_',dataset_name],target_data_path); % load feas and label
        yt = yt+1;
        
        t1 = clock;
        acc_matrix(i) = feval(func, xs,ys,xt,yt,param_opt); % domain adaptation method
        t2 = clock;
        
        time_matrix(i) = etime(t2,t1);
end
acc_matrix_percent = acc_matrix*100;
disp(['mean accuracy  = ' num2str(mean(acc_matrix_percent))]);
sumtime = sum(time_matrix);
disp(sumtime);    
save(['./res/' file_name '_acc_matrix.mat'],'acc_matrix_percent'); % save acc
save(['./res/' file_name '_time_matrix.mat'],'sumtime'); % save running time

% remove path
rmpath(genpath(path_data)); 
rmpath(genpath(path_util));
% clear var
clear acc_matrix acc_matrix_percent  sumtime xs ys xt yt param_opt
end




