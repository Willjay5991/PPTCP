% t-SNE visualization for Original, TCA, JDA, P2TCP feature
% by yjiedu@foxmail.com
clc, clear, close all;
%% add path
root = '../../';
path_opt.data = fullfile(root,'data'); % dataset
path_opt.util = fullfile(root, 'util'); % function
path_opt.p1 = '../../';

addpath(genpath(path_opt.data)); %add dataset
addpath(path_opt.util);
addpath(path_opt.p1);
%% load data
% dataset is Office31, task A -> D
%  method feature extraction: 
%       original:  non-adaptation
dataset_name = 'Office31';
[ ~,~,path ] = load_data(dataset_name);
src_name = 'D';
tar_name = 'A';

% xs:ns*d, ys: 1*ns
% xt:nt*d, yt: 1*nt  
source_data_path = [path, src_name, '.mat'];
[xs,ys] = feval(['load_',dataset_name],source_data_path); % load feas and label
ys = ys+1; % range is [1,C]
target_data_path = [path, tar_name, '.mat'];
[xt,yt] = feval(['load_',dataset_name],target_data_path); % load feas and label
yt = yt+1; % range is [1,C]

%       TCA:
%       JDA
%% visual orgin
if true
    X = [xs;xt];
    ns = size(xs,1);
    TSNE_visual(X,ns,'original')
end
%% visual TCA
if true
    if ~(exist('./res/TCA/domainS_proj_save.mat','file') ...
            && exist('./res/TCA/domainT_proj_save.mat','file'))
        opt.return_embededfeature=true; % save the embeded feature into /res/PGFL
        acc = TCA_api(xs,ys,xt,yt,opt);
    end
    % load
    dataS = load('./res/TCA/domainS_proj_save.mat');
    dataT = load('./res/TCA/domainT_proj_save.mat');
    xs_embeded = dataS.domainS_proj_save;
    xt_embeded = dataT.domainT_proj_save;
    X_embeded = [xs_embeded;xt_embeded];
    ns = size(xs_embeded,1);
    TSNE_visual(X_embeded,ns,'TCA');     
end
%% visual JDA
if true
    if ~(exist('./res/JDA/domainS_proj_save.mat','file') ...
            && exist('./res/JDA/domainT_proj_save.mat','file'))
        opt.return_embededfeature=true; % save the embeded feature into /res/PGFL
        acc = JDA_api(xs,ys,xt,yt,opt);
    end
    % load
    dataS = load('./res/JDA/domainS_proj_save.mat');
    dataT = load('./res/JDA/domainT_proj_save.mat');
    xs_embeded = dataS.domainS_proj_save;
    xt_embeded = dataT.domainT_proj_save;
    X_embeded = [xs_embeded;xt_embeded];
    ns = size(xs_embeded,1);
    TSNE_visual(X_embeded,ns,'JDA');     
end


%% visual P2TCP
if true
	 if ~(exist('./res/P2TCP/domainS_proj_save.mat','file') ...
        && exist('./res/P2TCP/domainT_proj_save.mat','file'))
       opt.return_embededfeature=true; % save the embeded feature into /res/PGFL
       opt.ds={512};
       opt.i=1;
       acc = SLPP_APM_api(xs,ys,xt,yt,opt);
   end
   % load
   dataS = load('./res/P2TCP/domainS_proj_save.mat');
   dataT = load('./res/P2TCP/domainT_proj_save.mat');
   xs_embeded = dataS.domainS_proj_save;
   xt_embeded = dataT.domainT_proj_save;
   X_embeded = [xs_embeded;xt_embeded];
   ns = size(xs_embeded,1);
   TSNE_visual(X_embeded,ns,'P2TCP');
end

%% remove path
rmpath(genpath(path_opt.data)); 
rmpath(path_opt.util);
rmpath(path_opt.p1);


function TSNE_visual(X,ns,name)
% X: n*d
X_embeded = tsne(X);
Y1 = repelem('source samples',ns,1);
Y2 = repelem('target samples',size(X_embeded,1)-ns,1);
Y = [Y1;Y2];
figure,gscatter(X_embeded(:,1),X_embeded(:,2),Y,'rb'),
set(gca,'XTick',[],'YTick',[]);
% title(name);
end




