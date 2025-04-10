
addpath(genpath('VBLab'))


clear
clc

seed = 1;
rng(seed)

[y_hat,y,x,p] = sim_lr_data(1000,[0.3,-1,2],0.5);
data = [y,x];

lm = fitlm(x,y,'Intercept',false);
tab_ml = [lm.Coefficients.Estimate;lm.RMSE];

% setting.fixed_rqmc = load('VBLab\VB\QBVI\rqmc\RQMC_LinRegr.mat','rqmc');

%%

clc
rng(seed)

setting.Prior.Mu        = ones(4,1);
setting.Prior.Sig       = 1;
setting.useHfunc        = 0;
setting.sampler         = 's';
setting.doCV            = 1;
setting.NgClip          = [100,1000];

setting.xi              = 1;

rng(seed)
pQBVI.out = QBVI(@h_func_LinRegr_QBVI,data,...
    'NumParams',4,...
    'Setting',setting,...
    'LearningRate',0.1,...
    'NumSample',25,...
    'MaxPatience',15000,...
    'MaxIter',500,...
    'GradWeight',0.4,...
    'WindowSize',30,...
    'SigInitScale',0.01,...
    'StepAdaptive',10000,...
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
tab_vi = [mu(1:3);fm(mu(4),s(4,1))];

array2table([tab_vi,tab_ml,p],'VariableNames',{'Estimated','ML','True'})


