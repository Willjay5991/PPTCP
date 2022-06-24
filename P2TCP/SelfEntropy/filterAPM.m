function [firstLabel,new_predLabel,new_prob,trustable, eta_,H_,predLabel_,maskMat_] = filterAPM(prob_all,APM,interval_M,iter,Xs,Xt,Ys,oldPredLabel, param_opt)
%% filter label with adaptative prototype memory, incremental filter
% d1<d2, keep firstLabel(x), otherwise drop it, 
% d1: max distance between x and centre class c in Mc, c=firstLabel(x)
% d2: min distance between x and centre class c in Mc, c=secondLabel(x)
% input : prob_all: size(nt,c)
%         APM.H: size(nt,1)
%         APM.predLabel: size(nt,1), predLabel of feature in M
%         APM.eta: size(1)
%         Xs: size(ns,d) embeded feature of source domain
%         Ys: size(ns,1) source label
%         Xt: size(nt,d) embeded feature of target domain
%         oldPredLabel: size(nt,1), predlabel of target in lastest
%         param_opt: struct
% output: firstLabel: predlabels before filter
%         new_predLabel:size(nt,1), predlabels after filter
%         new_prob: size(nt,1)
%         trustable: size(nt,1)
%         APM
% by yjiedu@foxmail.com

c_num = length(unique(Ys));
eta_mode = 'mat';  % 'mat'(default): eta_mat, 'number': eta
%% get firstLabel, secondLabel
[sort_Prob,order_Prob] = sort(-prob_all,2);
firstLabel  = order_Prob(:,1);
secondLabel = order_Prob(:,2);
firstProb   = -sort_Prob(:,1);
%% recover fixedlabel
firstLabel(APM.maskMat==1) = oldPredLabel(APM.maskMat==1);    
%% update APM
indx_updateM = iter;
if ((mod(indx_updateM, interval_M)==0) || (indx_updateM==1))
    [H_,eta_,eta_mat] = updateAPM(prob_all,firstLabel,APM, c_num, eta_mode, param_opt);  
    APM.H = H_;
    APM.eta =eta_;
    APM.eta_mat = eta_mat;
    APM.predLabel = firstLabel;
end
%% filter
predLabel_M = APM.predLabel;
eta = APM.eta;
H = APM.H;
if strcmp(eta_mode,'mat')
    indx_final = zeros(size(predLabel_M));
    for i=1:c_num
       indx= ((predLabel_M==i)&(H<=eta_mat(i)))|(APM.maskMat);
       indx_final = indx_final+indx;       
    end
    predLabel_M(~indx_final) = -1;
else
    predLabel_M((H>eta)&(~APM.maskMat)) = -1; % -1: if H>eta and not fixed label 
end

% prototype memory mode
PM_mode = param_opt.PM_mode; % 'ST'(default): PM include source and target; 'T': include target only
if strcmp(PM_mode,'ST')
    X = [Xs;Xt]; % (ns+nt,d)
    Y = [Ys;predLabel_M]; % (ns+nt,1)
else
    X = Xt;
    Y = predLabel_M;
end

dH_Q_Mt1 = zeros(size(firstLabel,1),1); % (nt,1), dH_Q_Mt1(i), the max distance between xi and firstlabelclass
dH_Q_Mt2 = dH_Q_Mt1;  % dH_Q_Mt2(i), the min distance between xi and secondlabClass
% firstLabel, secondLabel distance
for i =1:length(firstLabel) % for every sample
    if APM.maskMat(i)==1 % skip the fixedlabel
        continue
    end
    Mc_first = X(Y==firstLabel(i),:); % samples belong to firstlabel
    Mc_second = X(Y==secondLabel(i),:); % samples belong to secondlabel

    if isempty(Mc_first)
        dH_first = 2; % set the max if Mc_first not exist
    else
        dH_first = pdist2(X(i,:),Mc_first,'cosine'); % distance between xi and p, p is one vector of Mc_first
    end
    if isempty(Mc_second)
        dH_second = 2; % set the min if Mc_second not exist
    else
        dH_second = pdist2(X(i,:),Mc_second,'cosine');
    end 
    dH_Q_Mt1(i,1) = max(dH_first);
    dH_Q_Mt2(i,1) = min(dH_second);
end

mask_mat = dH_Q_Mt1 > dH_Q_Mt2;
new_predLabel = firstLabel; 
new_predLabel(mask_mat) = -1; % set untrust label as -1 
new_prob = firstProb; 
new_prob(mask_mat) = 0;
APM.maskMat = (new_predLabel~=-1) | APM.maskMat; % update fixedLabel mask matrix
eta_ = APM.eta;
H_ = APM.H;
predLabel_=APM.predLabel;
maskMat_ = APM.maskMat;
% print
etaFilter_indx = (H>eta)&(~APM.maskMat);
etaFilter_Percent = sum(etaFilter_indx)/length(etaFilter_indx);
trustable  = ~mask_mat;
percent = 1-sum(mask_mat)/length(mask_mat);
end

