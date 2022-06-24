function [H,eta,eta_mat] = updateAPM(prob_all, predLabel,APM, c_num,eta_mode,param_opt)
%% update APM
% input: prob_all: size(nt,c)
%        predLabel: size(nt,1)
%        APM.H: size(nt,1), self entropy 
%        APM.eta: size(1), threshold of self entropy
%        APM.predLabel: size(nt,1),
%        c_num: number class
%        eta_mode: 'mat': calc eta_mat; otherwise eta
%        param_opt: struct
% output: APM: new APM{ APM.eta, APM.H, APM.predLabel}
%         eta_mat:size(1,C)
% by yjiedu@foxmail.com
softmax_mod = param_opt.softmax_mod; % true(default), false
if softmax_mod 
    prob_all = softmax_row(prob_all); % softmax pro_all with its each row
end 
H = calc_selfEntropy(prob_all);
APM.H = H;
eta_mat = zeros(1,c_num);
eta = -inf;
if strcmp(eta_mode,'mat')
    eta_mat = classThreshold_selfEntropy(APM, predLabel,c_num);
else
    eta = threshold_selfEntropy(APM, predLabel,c_num);
end
end


%% calculate the self-entropy for target samples
% input: prob_tar: probabiltiy predicted label, size=(nt,C), C is number of classes 
% output: H: self-entropy, size = (nt,1)
% h(xt) = 1/log(Nc) * sum_{i=1...C}(prob_tar_{c_i}(xt)*log(prob_tar_{c_i}(xt)))
function H = calc_selfEntropy(prob_tar)
    Temp = prob_tar.*log(prob_tar+1e-6);
    H = -1.*sum(Temp,2)./log(size(prob_tar,2));
end


function [eta] = threshold_selfEntropy(APM,predLabel_tar,c_num)
%% calc eta: the threshold of self entropy
% APM process
% eta = max{min(H_c)| c in C}, H_c: sample self entropy for class c 
% input: APM.H: self-entropy,size=(nt,1)
%        APM.maskMat: fixed label: 1, otherwise:0
%        predLabel_tar: predicted target labels, size=(nt,1)
%        c_num: class number
% output: eta: threshold, size=(1)
H = APM.H;
maskMat = APM.maskMat;
eta = -inf;
for i =1:c_num
    indx = (predLabel_tar ==i)&(~maskMat);
    temp = min(H(indx));
    if temp > eta
        eta = temp;
    end
end
end

function [eta_mat] = classThreshold_selfEntropy(APM,predLabel_tar,c_num)
%% calc the threshold for each class
% input: APM
%        predLabel_tar: size(nt,1)
%        c_num
% output: eta_mat: size(1,C)
H = APM.H;
maskMat = APM.maskMat;
eta_mat = zeros(1,c_num);
ratio = 0.3;
for i =1:c_num
    Hc = H((predLabel_tar ==i)&(~maskMat));
    if ~isempty(Hc)
        sortedH = sort(Hc);
        threshold = sortedH(ceil(length(Hc)*ratio),1);
        eta_mat(i) = threshold;
    end
end
end

function Pro_softmax = softmax_row(Pro)
% func: softmax matrix Pro with its each row 
% pro (n,c)
% pro_softmax: (n,c)
% softmax(a)= exp(a)/sum(exp(a)), a is the column vector
Pro_softmax_T = softmax(Pro');
Pro_softmax = Pro_softmax_T';
end