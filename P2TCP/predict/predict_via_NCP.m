function [prob_all] = predict_via_NCP(Xs, Xt, Ys, num_class)
%% predict via NCP
% input: Xs: size(ns,d)
%        Xt: size(nt,d)
%        Ys: size(ns,1)
%        num_class: size(1)
% output: prob_all: size(nt,c)
% by yjiedu@foxmail.com

% distance to class means
classMeans = zeros(num_class,size(Xs,2));
for i = 1:num_class
    classMeans(i,:) = mean(Xs(Ys==i,:));
end
classMeans = L2Norm(classMeans); % size(c,d)
distClassMeans = EuDist2(Xt,classMeans); % size(nt,c)
expMatrix = exp(-distClassMeans);
probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
prob_all = probMatrix;
end