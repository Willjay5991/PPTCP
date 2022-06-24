function removepath4P2TCP(path_opt)
%% remove path
% by yjiedu@foxmail.com
names = fieldnames(path_opt);
for i = 1:length(names)
    rmpath(path_opt.(names{i}));
end
end