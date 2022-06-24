function [ Xs_pca, Xt_pca ] = do_PCA( Xs, Xt, opts )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   process src, tar domain data with PCA
X = double([Xs;Xt]);
P_pca = PCA(X,opts);
Xs = Xs*P_pca;
Xt = Xt*P_pca;
Xs_pca = L2Norm(Xs);
Xt_pca = L2Norm(Xt);
end

