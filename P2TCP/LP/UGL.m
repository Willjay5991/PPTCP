function [pred_scores] = UGL(W,a, num_examples, train_labels)
%% label propagation
% input: W weight matrix; num_example total sample num; train_labels
% output: pre_scores
% by yjiedu@foxmail.com

unique_labels = unique(train_labels);
num_c = length(unique_labels);

% make it symmetric
W = 0.5*(W + W');
D = diag(sum(W,2));
%S = diag(1./diag(sqrt(D)))*W*diag(1./diag(sqrt(D)));
S = D^(-0.5)*W*D^(-0.5);

% initial label matrix
label_mat = zeros(num_examples, num_c);
for ih = 1:num_c
    indx = find(train_labels==unique_labels(ih));
    label_mat(indx, unique_labels(ih))=1;
end

% propagate
pred_scores = (eye(size(a*S))-a*S)\label_mat;
end



