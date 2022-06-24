datasets = {'decaf6'};
funcs    = {'P2TCP_api'};
param_opt = struct;
param_opt.withSLPPs = {true};
param_opt.pred_modes = {'LP'}; % 'LP', 'NCP'
param_opt.filter_modes = {'APM'}; % 'APM'(default), 'confidence', 'nonfiltering'
param_opt.probModes = {'probDist'}; % % 'probLP':label propagation probability, 'probDist'(default): distance between qurey sample and target class center
param_opt.ds = {32}; 



% datasets = {'ImageCLEF','decaf6','Office31','OfficeHome'};
% funcs    = {'P2TCP_api','P2TCP_api','P2TCP_api','P2TCP_api'};
% param_opt = struct;
% param_opt.withSLPPs = {true, true, true, true};
% param_opt.pred_modes = {'LP', 'LP', 'LP', 'LP'}; % 'LP', 'NCP'
% param_opt.filter_modes = {'APM', 'APM', 'APM', 'APM'}; % 'APM'(default), 'confidence', 'nonfiltering'
% param_opt.probModes = {'probDist', 'probDist', 'probDist', 'probDist'}; % % 'probLP':label propagation probability, 'probDist'(default): distance between qurey sample and target class center
% param_opt.ds = {32, 32, 512, 512};  

