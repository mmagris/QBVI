
seed = 1;


%% FOR LINEAR REGRESSION



S = 100;
Par = 4;

rng(seed)
rqmc = normrnd(0,1,S,4);
save('VBLab\VB\QBVI\rqmc\RQMC_LinRegr.mat','rqmc')


%% FOR LOGISTIC REGRESSION

S = 100;
Par = 4;

rng(seed)
rqmc = normrnd(0,1,S,4);
save('VBLab\VB\QBVI\rqmc\RQMC_LogRegr.mat')