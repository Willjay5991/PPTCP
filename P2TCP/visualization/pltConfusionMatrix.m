% the visualization of the confusion matrix
% by yjiedu@foxmail.com
clear, clc;
close all;

STs = {'AW','DA'};
methods = {'1NN', 'P2TCP'};
% for i=1:length(STs)
% 	pltConfusionMat()
% 	
% end
%% classes for imageclef
% xValues = {'airplane','bike','bird','boat','bottle','bus','car','dog','horse','monitor','motorbike','people'};
% yValues = xValues;
%% classed for Office31
xValues = {'backpack', 'bike', 'bikehelmet', 'bookcase', 'bottle', ...
           'calculator', 'desktopcomputer', 'deskchair', 'desklamp', ...
           'filecabinet', 'headphones',	'keyboard',	'laptopcomputer', 'lettertray', ...
           'mobilephone', 'monitor', 'mouse', 'mug', 'papernotebook', 'pen', ...
           'phone',	'printer', 'projector',	'punchers', 'ringbinder', 'ruler', ...
           'scissors', 'speaker', 'stapler', 'tapedispenser', 'trashcan'};
yValues = xValues;
for i=1:length(STs)
    for j=1:length(methods)
        pltConfusionMat(STs{i}, methods{j}, xValues, yValues);
    end
end


function pltConfusionMat(ST, method, xValues, yValues)
    name = 'Normalized confusion matrix';
    load(['res/' ST '/' method '-' 'predLabel.mat']); % load 'pred_labels': size=(n,1)
    load(['res/' ST '/' method '-' 'TrueLabel.mat']); % load 'test_labels'  size=(n,1)
    test_mat = oneHot(test_labels'); % n*c
    pred_mat = oneHot(pred_labels'); % n*c
%     figure(),
%     plotconfusion(test_mat', pred_mat'),
%     title([method ':' ST]);
    [~,cMat,~,~] = confusion(test_mat', pred_mat');
    cMat = norm4Row(cMat);
    figure()
    h = heatmap(xValues, yValues, cMat);
    h.CellLabelFormat = '%0.2f';
    h.CellLabelColor = 'none';
    title(name)
    xlabel('Predicted Label');
    ylabel('True Label');
end

%% vector to oneHotMat
function oneHotMat = oneHot(vector)
    L = length(unique(vector));
    eyeMat = eye(L);
    oneHotMat = zeros(size(vector,2),L);
    for i= 1:size(vector,2)
        oneHotMat(i,:) = eyeMat(vector(i),:);
    end
end

%% normlize along column
function matNormRow = norm4Row(mat)
    matNormRow = mat./repmat(sum(mat,2),1,size(mat,2));
end
