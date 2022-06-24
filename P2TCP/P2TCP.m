% =====================================================================
% Code for the paper:
% Yongjie Du, Unsupervised Domain Adaptation via Progressive Positioning of
% Target-Class Prototypes
% By Yongjie Du, yjiedu@foxmail.com
% =====================================================================
function [acc, acc_per_class, pseudoLabels] = P2TCP(domainS_features,domainS_labels,domainT_features,domainT_labels,param_opt,return_embededfeature)
%% descripation
% input: domainS_features, domainT_features: n*d
%        domainS_labels, domainT_labels: 1*n
% output: pseudoLabels,

num_iter = param_opt.T;
options.NeighborMode='KNN';
options.WeightMode = 'HeatKernel';
options.k = 30;
options.t = 1;
options.alpha = 1;
options.ReducedDim = param_opt.d;
options.init_mode = 'KNN'; % KNN(default), NCP_SPL
options.W_mode = '0_1_mode'; % '0_1_mode'(default): wij=1 or 0, '1_sim_mode': 1 or similarity
options.pred_mode = param_opt.pred_mode; % LP(defult), SVM, OfficialSVM, selfEntropy, dist, NCP_SPL
options.filter_mode = param_opt.filter_mode; % APM(default), confidence
disp(options);

num_class = length(unique(domainS_labels));

%% initialize pseudo labels 
if strcmp(options.init_mode,'KNN')  % initialize pseudo labels with KNN
    I = eye(size(domainS_features,2));
    [predLabels_init,~,acc_initi] = my_kernel_knn(I,domainS_features,domainS_labels',domainT_features,domainT_labels');
end
disp(['initial acc:', num2str(acc_initi)]),
pseudoLabels = predLabels_init';

%% initialize with pseudolabel and labels
if strcmp(options.W_mode, '0_1_mode')
    W = constructW1([domainS_labels,pseudoLabels]);
end
% looping
fprintf('d=%d\n',options.ReducedDim);
%% initial APM
APM.eta = -inf;
APM.H = [];
APM.predLabel = predLabels_init;
APM.maskMat = zeros(size(domainT_features,1),1); % fixed label: 1, otherwise:0
interval_M = 1;
%% 
acc = zeros(num_iter,1);
acc_per_class = zeros(num_iter,num_class);
for iter = 1:num_iter
    if param_opt.withSLPP
       %% project data into a latent space
        P = LPP([domainS_features;domainT_features],W,options);
        [domainS_proj, domainT_proj] = normalization(domainS_features, domainT_features, P);
    else
        domainS_proj = domainS_features;  domainT_proj = domainT_features;
    end
  	 
    %% predict target data:  prob:size=1*n, predLabels:size=1*n 
    if strcmp(options.pred_mode,'LP')
        [prob_all] = predict_via_LP_APM(domainS_proj,domainT_proj,domainS_labels',pseudoLabels',param_opt); % SLPP+LP
    elseif strcmp(options.pred_mode,'NCP')
        [prob_all] = predict_via_NCP(domainS_proj,domainT_proj,domainS_labels', num_class);
    end
	
 	%% filter labels
    if strcmp(options.filter_mode,'APM')  % filter with APM
        [predLabels,filter_predLabels,~, ~,eta_,H_,predLabel_,maskMat_] = filterAPM(prob_all,APM,interval_M,iter,domainS_proj,domainT_proj,domainS_labels',pseudoLabels', param_opt);
        APM.eta = eta_;
        APM.H = H_;
        APM.predLabel = predLabel_;
        APM.maskMat = maskMat_;
%         printH(H_,predLabels,domainT_labels',iter,num_class);
    elseif strcmp(options.filter_mode,'nonfiltering')
        [~,predLabels] = max(prob_all'); % 1*n
        predLabels = predLabels';
        filter_predLabels = predLabels;
    end
    pseudoLabels = filter_predLabels';    % (nt,1)-->(1,nt)    
    
    %% rebulid W
    if strcmp(options.W_mode, '0_1_mode')
        W = constructW1([domainS_labels,pseudoLabels]);
    end
    %% calculate ACC
    acc(iter) = sum(predLabels'==domainT_labels)/length(domainT_labels);
    disp(['iter:',num2str(iter), ' acc:', num2str(acc(iter))]),
    
end
if return_embededfeature % for visualization
    % save prediction
    predictionPath = ['./res/PPTCP/' param_opt.stNames];
    mkdir(predictionPath);
    pred_labels = predLabels;
    test_labels = domainT_labels';
    save([predictionPath, '/PPTCP-PredLabel.mat'],'pred_labels');
    save([predictionPath, '/PPTCP-TrueLabel.mat'],'test_labels');
    % save feature
    [domainS_proj_save, domainT_proj_save] = normalization(domainS_features, domainT_features, P);
    save([predictionPath '/domainS_proj_save.mat'],'domainS_proj_save');
    save([predictionPath, '/domainT_proj_save.mat'],'domainT_proj_save');
end
end



%% z = z - z_mean
%	 \hat z = L2Norm(z)
function [domainS_proj, domainT_proj] = normalization(domainS_features, domainT_features, P)
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
end


%% initialize pseudo-labels with KNN
%% the square of Euclid distance
% dist(k1,D1) = k1^2 + d1^2 - 2*k1*d1
function [prediction,prob, accuracy] = my_kernel_knn(M, Xr, Yr, Xt, Yt)
% input: M:size(d,d)
%        Xr: size(ns,d)
%        Yr: size(ns,1)
%        Xt: size(nt,1)
%        Yt: size(nt,d)
    dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) ...
        + repmat(diag(Xt*M*Xt')',length(Yr),1)...
        - 2*Xr*M*Xt';
    % norm dist
    dist_norm = dist./repmat(sum(dist),length(Yr),1);  % norm with respect to column
    [prob, minIDX] = min(dist_norm);
    prediction = Yr(minIDX);
    % select
%     [pseudoLabels,~] = select_predictedlabel(5, 3, prob, prediction', num_class);
%     prediction = pseudoLabels';
    accuracy = sum( prediction==Yt ) / length(Yt);
end 
