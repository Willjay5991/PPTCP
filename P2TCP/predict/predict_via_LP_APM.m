function [prob_all] = predict_via_LP_APM(Xs,Xt,Ys,predLabels,param_opt)
%% predict label with label propagation
% input: Xs: size(ns,d)
%        Ys: size(ns,1)
%        Xt: size(nt,d)
%        pred_Yt: size(nt,1)
%        APM: struct, APM.H: size(nt,1); APM.eta:size(1);
%        APM.predLabel:size(nt,1)
% output: prob: size(nt,1)
%         predLabel: size(nt,1)
%         prob_all: size(nt,c)
% by yjiedu@foxmail.com

%% parameter
k = 15;
a = 0.9;
num_examples = size(Xs,1)+size(Xt,1);
num_clus = length(unique(Ys));
W = constructW(Xs,Xt,k);  % default dist

% unsupervised graph learning
pred_scores = UGL(W,a,num_examples,Ys);  % (ns+nt, c)
% predict labels with softmax(F)
pred_scores = pred_scores(length(Ys)+1:end, :); % (nt,c)
pred_scores_norm = pred_scores./repmat(sum(pred_scores,2),1,size(pred_scores,2)); % norm with repect to row

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
probMode = param_opt.probMode;   % 'probLP':label propagation probability, 'probDist'(default): distance between qurey sample and target class center
                     % probMax: max(probLP, probdist)
withSrcCenters = param_opt.withSrcCenter; % true(default): include src classcenters when calc the tar classcenters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
if ~strcmp(probMode,'probLP')
    % cluster information
    [~,predLabel_tar] = max(pred_scores_norm,[],2);
    ClassCenters_tar = zeros(length(unique(Ys)),size(Xt,2));
    ClassCenters_src = ClassCenters_tar;
    % src domain classcenters
    for c = 1:length(unique(Ys))
        ClassCenters_src(c,:) = mean(Xs(Ys==c,:));
    end
    for c = 1:length(unique(Ys))
        if sum(predLabel_tar==c)==0
            ClassCenters_tar(c,:) = mean(Xs(Ys==c,:)); % if miss target label, replace with corresponding src center
        else
            if withSrcCenters 
                Xt_c = [Xt(predLabel_tar==c,:); ClassCenters_src(c,:)];
            else
                Xt_c = Xt(predLabel_tar==c,:);
            end
            ClassCenters_tar(c,:) = mean(Xt_c); 
        end
    end
    ClassCenters_tar = L2Norm(ClassCenters_tar);
    distTarCenter = EuDist2(Xt, ClassCenters_tar);
    expMatrix = exp(-distTarCenter);
    probMatrix = expMatrix./repmat(sum(expMatrix,2)+1e-6,1,size(expMatrix,2));
end


if strcmp(probMode,'probLP')
    prob_all = pred_scores_norm; % probmax
elseif strcmp(probMode,'probDist')
    prob_all =  probMatrix; % probDist
elseif strcmp(probMode,'probMax')
    prob_all = max(pred_scores_norm,probMatrix); % max()
end   

end

% construct W with structural informaton only
function W = constructW(train_fea,test_fea,k)
X = [train_fea',test_fea']; % X: size=(d, (ns+nt)) 
W = zeros(size(X,2)); % size = (ns+nt)*(ns+nt)
% k nearest neighbor
[idx, dist] = knnsearch(X', X', 'distance','cosine','k', k);
for ix = 1:size(X,2)
    W(ix, idx(ix,:)) = 1-dist(ix,:);
    W(ix,ix) = 0;
end
end

