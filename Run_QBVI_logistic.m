
addpath(genpath('VBLab'))
rmpath(genpath('VBLab/VB/MGVB'))
rmpath(genpath('VBLab/VB/EMGVB'))


clear
clc

seed = 1;
rng(seed)

par = [0.1;0.3;-1;2];
x = rand(1000,3);
[~,y_prob] = predict_glm(x,par(2:end),par(1));

y = binornd(1,y_prob);

data = [y,x];

tab_ml = glmfit(x,y,'binomial');

setting.fixed_rqmc = load('VBLab\VB\QBVI\rqmc\RQMC_LogRegr.mat','rqmc');

%%

clc
rng(seed)

setting.Prior.Mu        = ones(4,1);
setting.Prior.Sig       = 5;
setting.useHfunc        = 0;
setting.sampler         = 's';
setting.doCV            = 1;
setting.NgClip          = [100,5000];

setting.xi              = 4;

rng(seed)
pQBVI.out = QBVI(@h_func_LogRegr_QBVI,data,...
    'NumParams',4,...
    'Setting',setting,...
    'LearningRate',0.08,...
    'NumSample',50,...
    'MaxPatience',15000,...
    'MaxIter',2000,...
    'GradWeight',0.4,...
    'WindowSize',30,...
    'SigInitScale',0.01,...
    'StepAdaptive',1000,...
    'GradientMax',0,...
    'GradClipInit',10,...
    'SaveParams',true,...
    'Verbose',1,...
    'LBPlot',1);

%%


use = pQBVI.out.Post;

mu = use.mu;
s  = use.Sig;

fm = @(m,s) exp(m+s/2);
tab_vi = mu;

array2table([tab_vi,tab_ml,par],'VariableNames',{'QBVI','ML','True'})


