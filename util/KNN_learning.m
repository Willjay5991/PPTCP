function [ accuracy ] = KNN_learning( Xr,Yr,Xt,Yt,opt )
%KNN_LEARNING Summary of this function goes here
% Detailed explanation goes here
% 1NN learning(without domain adaptation)
% the square of Euclid distance
%    dist(k1,D1) = k1^2 + d1^2 - 2*k1*d1
% by yjiedu@foxmail.com
ifSaveConfusionMatrix = true; % false (default)

%% L2Norm 
Xr = L2Norm(Xr);
Xt = L2Norm(Xt);
%% PCA
opts.ReducedDim = 256; %128
[Xr,Xt] = do_PCA(Xr,Xt,opts);      

M = eye(size(Xr,2));
% normalization via norm2
[Xr,Xt] = normalize_via_norm2(Xr,Xt);
dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) ...
        + repmat(diag(Xt*M*Xt')',length(Yr),1)...
        - 2*Xr*M*Xt';
[~, minIDX] = min(dist);
prediction = Yr(minIDX);
accuracy = sum( prediction==Yt ) / length(Yt); 
disp(['acc=', num2str(accuracy)])
%% save confusion matrix
if ifSaveConfusionMatrix
	pred_labels = prediction';
	test_labels = Yt';
	save(['./res/' '1NN-PredLabel.mat'],'pred_labels');
	save(['./res/' '1NN-TrueLabel.mat'],'test_labels');
end

end

%% normalize data via norm2
function [Xr_norm2,Xt_norm2] = normalize_via_norm2(Xr,Xt)
% (x1,x2,x3,...,xn)/sqrt(x1^2+x2^2+x3^2+...+xn^2)
Xr_norm2 = Xr./repmat(sqrt(sum(Xr.^2,2)),1,size(Xr,2));
Xt_norm2 = Xt./repmat(sqrt(sum(Xt.^2,2)),1,size(Xt,2));
end