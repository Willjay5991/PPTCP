function [acc] = P2TCP_api(xs,ys,xt,yt,param_opt)
% P2TCP_api Summary of this function goes here
%   Detailed explanation goes here
% the api for P2TCP
% by yjiedu@foxmail.com
path_opt = addpath4P2TCP(); % add path


if ~isfield(param_opt,'ds')
    param_opt.d = 128; % 128(default)
else
	  param_opt.d = param_opt.ds{param_opt.i};
end

if ~isfield(param_opt,'Ts')
    param_opt.T=8; % 8(default)
else
	  param_opt.T = param_opt.Ts{param_opt.i};
end

if ~isfield(param_opt,'pred_modes')
    param_opt.pred_mode = 'LP';  % SP_NCP, LP(defult), SVM, OfficialSVM, selfEntropy, dist, NCP_SPL
else
    param_opt.pred_mode = param_opt.pred_modes{param_opt.i};
end

if ~isfield(param_opt,'filter_modes')
    param_opt.filter_mode = 'APM';  % % APM(default), confidence
else
    param_opt.filter_mode = param_opt.filter_modes{param_opt.i};
    if strcmp(param_opt.filter_mode, 'nonfiltering')
        param_opt.T = 1;
    end
end

if ~isfield(param_opt,'softmax_mods')
    param_opt.softmax_mod = true;  % true(default), false, whether apply softmax to the prob matrix
else
    param_opt.softmax_mod = param_opt.softmax_mods{param_opt.i};
end

if ~isfield(param_opt,'PM_modes')
    param_opt.PM_mode = 'T';  % 'ST': PM include source and target; 'T'(default): include target only
else
    param_opt.PM_mode = param_opt.PM_modes{param_opt.i};
end

if ~isfield(param_opt,'probModes')
    param_opt.probMode = 'probDist';  % probLP, probDist(default), probMax
else
    param_opt.probMode = param_opt.probModes{param_opt.i};
end

if ~isfield(param_opt,'withSrcCenters')
    param_opt.withSrcCenter = true;  % true(default), false
else
    param_opt.withSrcCenter = param_opt.withSrcCenters{param_opt.i};
end

if ~isfield(param_opt, 'withSLPPs')
    param_opt.withSLPP = true; % true(default): with SLPP
else
    param_opt.withSLPP = param_opt.withSLPPs{param_opt.i};
end

if ~isfield(param_opt,'return_embededfeature')
    return_embededfeature = false; % ·µ»Ø´æ´¢embeded feature µ½./res/
else
    return_embededfeature = param_opt.return_embededfeature;
end

%% L2Norm 
xs = L2Norm(xs);
xt = L2Norm(xt);
%% PCA
opts.ReducedDim = param_opt.d; %128, 256(default)
[xs,xt] = do_PCA(xs,xt,opts);  
param_opt.d = size(xs,2); % important, after do_PCA size(xs,2) may not equal param_opt.d;
disp(param_opt);
fprintf(' PCA dim=%d \n SLPP dim=%d \n iteration=%d ', opts.ReducedDim, param_opt.d, param_opt.T);
[accs,~,~] = P2TCP(xs,ys,xt,yt,param_opt,return_embededfeature);
acc = accs(end);


removepath4P2TCP(path_opt); % remove path
end