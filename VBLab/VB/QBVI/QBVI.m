classdef QBVI < VBayesLab
    %MVB Summary of this class goes here
    %   Detailed explanation goes here

    properties
        GradClipInit       % If doing gradient clipping at the beginning
    end

    methods
        function obj = QBVI(mdl,data,varargin)
            %MVB Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method        = 'QBVI';
            obj.GradWeight    = 0.4;    % Small gradient weight is better
            obj.GradClipInit  = 0;      % Sometimes we need to clip the gradient early

            % Parse additional options
            if nargin > 2
                paramNames = {'NumSample'             'LearningRate'       'GradWeight'      'GradClipInit' ...
                    'MaxIter'               'MaxPatience'        'WindowSize'      'Verbose' ...
                    'InitMethod'            'StdForInit'         'Seed'            'MeanInit' ...
                    'SigInitScale'          'LBPlot'             'GradientMax' ...
                    'NumParams'             'DataTrain'          'Setting'         'StepAdaptive' ...
                    'SaveParams'};
                paramDflts = {obj.NumSample            obj.LearningRate    obj.GradWeight    obj.GradClipInit ...
                    obj.MaxIter              obj.MaxPatience     obj.WindowSize    obj.Verbose ...
                    obj.InitMethod           obj.StdForInit      obj.Seed          obj.MeanInit ...
                    obj.SigInitScale         obj.LBPlot          obj.GradientMax  ...
                    obj.NumParams            obj.DataTrain       obj.Setting       obj.StepAdaptive ...
                    obj.SaveParams};

                [obj.NumSample,...
                    obj.LearningRate,...
                    obj.GradWeight,...
                    obj.GradClipInit,...
                    obj.MaxIter,...
                    obj.MaxPatience,...
                    obj.WindowSize,...
                    obj.Verbose,...
                    obj.InitMethod,...
                    obj.StdForInit,...
                    obj.Seed,...
                    obj.MeanInit,...
                    obj.SigInitScale,...
                    obj.LBPlot,...
                    obj.GradientMax,...
                    obj.NumParams,...
                    obj.DataTrain,...
                    obj.Setting,...
                    obj.StepAdaptive,...
                    obj.SaveParams] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
            end

            % Check if model object or function handle is provided
            if (isobject(mdl)) % If model object is provided
                obj.Model = mdl;
                obj.ModelToFit = obj.Model.ModelName; % Set model name if model is specified
            else % If function handle is provided
                obj.HFuntion = mdl;
            end

            % Main function to run MGVB
            obj.Post   = obj.fit(data);
        end

        %% VB main function
        function Post = fit(obj,data)

            % Extract model object if provided
            if (~isempty(obj.Model))
                model           = obj.Model;
                d_theta         = model.NumParams;      % Number of parameters
            else  % If model object is not provided, number of parameters must be provided
                if (~isempty(obj.NumParams))
                    d_theta = obj.NumParams;
                else
                    error('Number of model parameters have to be specified!')
                end
            end

            % Extract sampling setting
            std_init        = obj.StdForInit;
            eps0            = obj.LearningRate;
            S               = obj.NumSample;
            ini_mu          = obj.MeanInit;
            window_size     = obj.WindowSize;
            max_patience    = obj.MaxPatience;
            momentum_weight = obj.GradWeight;
            init_scale      = obj.SigInitScale;
            stepsize_adapt  = obj.StepAdaptive;
            max_iter        = obj.MaxIter;
            lb_plot         = obj.LBPlot;
            max_grad        = obj.GradientMax;
            max_grad_init   = obj.GradClipInit;
            hfunc           = obj.HFuntion;
            setting         = obj.Setting;
            verbose         = obj.Verbose;
            save_params     = obj.SaveParams;

            setting.d_theta = d_theta;
            [setting, useHfunc] = check_setting(setting);

            % Store some variables at each iteration
            ssave = create_iter_struct(save_params,max_iter,setting);

            % Initialization
            iter        = 0;
            patience    = 0;
            stop        = false;
            LB_smooth   = zeros(1,max_iter+1);
            LB          = zeros(1,max_iter+1);

            if isempty(ini_mu)
                mu = normrnd(0,std_init,d_theta,1);
            else
                mu = ini_mu;
            end


            % Initialize Sig_inv
            u       = ones(d_theta,1);
            n_par   = d_theta+d_theta;
            Sig_inv = u./init_scale;
            k = setting.xi;

            % Functions, for Sig full or diagonal
            log_pdf      = @(theta,mu,Sig_inv) -d_theta/2*log(2*pi)+1/2*sum(log(Sig_inv))-1/2*(theta-mu)'*((theta-mu).*Sig_inv);
            fun_gra_log_q_Sig = @(aux,Sig_inv) Sig_inv-Sig_inv.*aux.*(aux).*Sig_inv;


            if setting.doCV == 1
                fun_cv = @(A,B) control_variates(A,B,S,1);
            else
                fun_cv = @(A,B) control_variates(A,B,S,0);
            end

            c12                         = zeros(1,n_par);
            gra_log_q_lambda            = zeros(S,n_par);
            grad_log_q_h_function       = zeros(S,n_par);
            grad_log_q_h_function_cv    = zeros(S,n_par);
            lb_log_h                    = zeros(S,1);

            if save_params
                llh_s = zeros(S,1);
                log_q_lambda_s = zeros(S,1);
            end

            Sig = get_Sig(Sig_inv);

            theta_all = f_sampler(mu,Sig,S,setting);

            for s = 1:S
                % Parameters in Normal distribution
                theta = theta_all(:,s);

                [h_theta,llh] = hfunc(data,theta,setting);

                % Log q_lambda
                log_q_lambda = log_pdf(theta,mu,Sig_inv);

                % h function
                h_function = h_theta - log_q_lambda;

                if useHfunc
                    f = h_function;
                else
                    f = llh;
                end

                if save_params
                    llh_s(s,1) = llh;
                    log_q_lambda_s(s,1) = log_q_lambda;
                end

                % To compute the lowerbound
                lb_log_h(s) = h_function;

                aux                           = (theta-mu);
                gra_log_q_mu                  = aux;
                gra_log_q_Sig                 = fun_gra_log_q_Sig(aux,Sig_inv);
                gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
            end

            c12 = fun_cv(grad_log_q_h_function,gra_log_q_lambda);
            Y12 = mean(grad_log_q_h_function_cv)';
            Y12 = grad_clipping(Y12,max_grad_init);

            [gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig_inv);
            [gradLB_mu,gradLB_iSig] = Ng_clipping(gradLB_mu,gradLB_iSig,setting.NgClip(1));

            gradLB_iSig_momentum   = gradLB_iSig;
            gradLB_mu_momentum     = gradLB_mu;

            LB0 = mean(lb_log_h);
            if verbose ~= 0
                disp(['Iter: 0000 |LB: ', num2str(LB0)])
            end

            % Prepare for the next iterations
            mu_best = mu;
            Sig_inv_best = Sig_inv;

            while ~stop

                iter = iter+1;
                if iter>stepsize_adapt
                    stepsize = eps0*stepsize_adapt/iter;
                else
                    stepsize = eps0;
                end

                if k == 0
                    Sig_inv = Sig_inv + stepsize*gradLB_iSig_momentum;
                else                                      
                    Sig_inv = Sig_inv.*exp(stepsize/k*Sig.*gradLB_iSig_momentum);
                end

                Sig = get_Sig(Sig_inv);

                mu = mu + stepsize*Sig.*gradLB_mu_momentum;

                if save_params
                    llh_s = zeros(S,1);
                    log_q_lambda_s = zeros(S,1);
                end

                theta_all = f_sampler(mu,Sig,S,setting);

                for s = 1:S
                    % Parameters in Normal distribution
                    theta = theta_all(:,s);

                    [h_theta,llh] = hfunc(data,theta,setting);

                    % log q_lambda
                    log_q_lambda = log_pdf(theta,mu,Sig_inv);

                    % h function
                    h_function = h_theta - log_q_lambda;

                    if useHfunc
                        f = h_function;
                    else
                        f = llh;
                    end

                    if save_params
                        llh_s(s,1) = llh;
                        log_q_lambda_s(s,1) = log_q_lambda;
                    end

                    % To compute the lowerbound
                    lb_log_h(s) = h_function;

                    aux                           = (theta-mu);
                    gra_log_q_mu                  = aux;
                    gra_log_q_Sig                 = fun_gra_log_q_Sig(aux,Sig_inv);
                    gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                    grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
                end

                c12 = fun_cv(grad_log_q_h_function,gra_log_q_lambda);
                Y12 = mean(grad_log_q_h_function_cv)';
                Y12 = grad_clipping(Y12,max_grad);

                [gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig_inv);
                
                [gradLB_mu,gradLB_iSig] = Ng_clipping(gradLB_mu,gradLB_iSig,setting.NgClip(end));

                gradLB_iSig_momentum = momentum_weight*gradLB_iSig_momentum+(1-momentum_weight)*gradLB_iSig;
                gradLB_mu_momentum   = momentum_weight*gradLB_mu_momentum+(1-momentum_weight)*gradLB_mu;

                % Lower bound
                LB(iter) = mean(lb_log_h);



                % Smooth the lowerbound and store best results
                if iter>window_size
                    LB_smooth(iter-window_size) = mean(LB(iter-window_size:iter));
                    if LB_smooth(iter-window_size)>=max(LB_smooth(1:iter-window_size))
                        mu_best  = mu;
                        Sig_inv_best = Sig_inv;
                        patience = 0;
                    else
                        patience = patience + 1;
                    end
                end

                if (patience>max_patience)||(iter>max_iter)
                    stop = true;
                end

                % Display training information
                print_training_info(verbose,stop,iter,window_size,LB,LB_smooth)

                % If users want to save variational mean, var-cov matrix and ll, log_q at each iteration
                if(save_params)
                    ssave = write_iter_struct(ssave,save_params,iter,mu,llh_s,log_q_lambda_s,Sig_inv);
                end

            end

            LB_smooth = LB_smooth(1:(iter-window_size-1));
            LB = LB(1:iter-1);

            % Store output
            Post.LB0        = LB0;
            Post.LB         = LB;
            Post.LB_smooth  = LB_smooth;
            [Post.LB_max,Post.LB_indx] = max(LB_smooth);
            Post.mu         = mu_best;
            [~,Post.ll]     = hfunc(data,Post.mu,setting); %ll computed in posterior mean
            Post.Sig_inv    = Sig_inv_best;
            Post.Sig        = get_Sig(Sig_inv);

            Post.Sig2   = Post.Sig;

            Post.setting = setting;
            if(save_params)
                Post.iter = ssave;
            end

            % Plot lowerbound
            if(lb_plot)
                obj.plot_lb(LB_smooth);
            end
        end

    end
end


function[ssave] = create_iter_struct(save_params,max_iter,setting)

ssave = struct();
d_theta = setting.d_theta;

if(save_params)
    ssave.mu    = zeros(max_iter,d_theta);
    ssave.ll    = zeros(max_iter,1);
    ssave.logq = zeros(max_iter,1);

    ssave.SigInv = zeros(max_iter,d_theta);

end
end

function[ssave] = write_iter_struct(ssave,save_params,iter,mu,llh_s,log_q_lambda_s,Sig_inv)

if(save_params)
    ssave.mu(iter,:)     = mu;
    ssave.ll(iter,:)     = mean(llh_s);
    ssave.logq(iter,:)   = mean(log_q_lambda_s);
    ssave.SigInv(iter,:) = Sig_inv(:)';
end
end

function Y12 = grad_clipping(Y12,max_grad)
if max_grad>0
    grad_norm = norm(Y12);
    norm_gradient_threshold = max_grad;
    if grad_norm > norm_gradient_threshold
        Y12 = (norm_gradient_threshold/grad_norm)*Y12;
    end
end
end

function[c] = control_variates(A,B,S,do)
if do
    c = (mean(A.*B)-mean(A).*mean(B))./var(B)*S/(S-1);
else
    c = 0;
end
end

function[] = print_training_info(verbose,stop,iter,window_size,LB,LB_smooth)
if verbose == 1
    if iter> window_size

        LBimporved = LB_smooth(iter-window_size)>=max(LB_smooth(1:iter-window_size));

        if LBimporved
            str = '*';
        else
            str = ' ';
        end

        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter-window_size)),str])
    else
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
    end
end

if verbose == 2 && stop == true
    if iter> window_size
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter-window_size))])
    else
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
    end
end
end


function[gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig_inv)

useHfunc    = setting.useHfunc;
Sig0_type   = setting.Sig0_type;
d_theta     = setting.d_theta;
mu0         = setting.mu0;
Sig_inv_0   = setting.iSig0;

if useHfunc
    C_s     = 0;
    C_mu    = 0;
else
    if Sig0_type == 1
        C_s     = diag(Sig_inv_0)-Sig_inv;
        C_mu    = Sig_inv_0*(mu0-mu);
    else
        C_s     = Sig_inv_0-Sig_inv;
        C_mu    = Sig_inv_0.*(mu0-mu);
    end
end

gradLB_mu       = C_mu + Sig_inv.*Y12(1:d_theta);
gradLB_iSig     = C_s  + Y12(d_theta+1:end);

end

function[Sig] = get_Sig(Sig_inv)

Sig = 1./Sig_inv;

end

function[setting, useHfunc] = check_setting(setting)

if ~isfield(setting,'doCV')
    warning('Missing doCV field in setting. Setting it to "true".')
    setting.doCV = 1;
else
    if ~(setting.doCV == 1 || setting.doCV == 0)
        error('Invalid setting.doCV field in setting.')
    end
end

if  ~isfield(setting,'useHfunc')
    warning('Missing sampler field in setting. Setting it to "true".')
    setting.useHfunc = 1;
else
    if ~(setting.useHfunc == 1 || setting.useHfunc == 0)
        error('Invalid setting.useHfunc field in setting.')
    end
end

if  ~isfield(setting,'sampler')
    warning('Missing sampler field in setting. Setting it to "n".')
    setting.sampler = 'n';
else
    if ~(setting.sampler == 'n' || setting.sampler == 's' || setting.sampler == 'f')
        error('Invalid setting.sampler field in setting.')
    end
end

if  ~isfield(setting,'xi')
    warning('Missing xi field in setting. Setting it to "0".')
    setting.xi = 0;
else
    if setting.xi < 0
        error('Invalid setting.xi. Must be a positive scalar.')
    end
end

% Check prior Sigma

dim = size(setting.Prior.Sig);
if dim(1)>1 && dim(2)>1
    error('Prior Sigma cannot be a matrix. Use a scalar or a vector.')
end

if dim(1)<dim(2)
    setting.Prior.Sig = setting.Prior.Sig';
end
N_Sig = numel(setting.Prior.Sig);

if N_Sig == 1
    Sig0_type = 1; %scalar
else
    Sig0_type = 2; %vector
end

Sig_inv_0 = 1./setting.Prior.Sig;


% Check prior Mu
dim = size(setting.Prior.Mu);
if dim(1)>1 && dim(2)>1
    error('Prior Mu cannot be a matrix. Use a vector.')
end

if dim(1)<dim(2)
    setting.Prior.Mu = setting.Prior.Mu';
end
N_Mu = numel(setting.Prior.Mu);

% Check Mu and Sig
if Sig0_type == 2 && (N_Mu ~= N_Sig)
    error('The numer of elements in prior Sig (vector) does not match the number of elements in prior Mu.')
end


% Check Natural Gradient Clippling

if  ~isfield(setting,'NgClip')
    warning('Missing NgClip field in setting. Setting it to "disabled".')
    setting.NgClip = '0';
else
    if numel(setting.NgClip)>2 || any(setting.NgClip<0)
        error('Invalid setting.NgClip: at most two elements, both positive.')
    end
end

useHfunc            = setting.useHfunc;
setting.iSig0       = Sig_inv_0;
setting.Sig0_type   = Sig0_type;
setting.mu0         = setting.Prior.Mu;

end


function[theta_all] = f_sampler(mu,Sig,S,setting)

d_theta = numel(mu);

switch setting.sampler
    case 's' % Sobol sequence
        rqmc = utils_normrnd_qmc(S,d_theta);
    case 'n' % Normal(0,1)
        rqmc = randn(S,d_theta);
    case 'f' % Fixed values from file - for debugging only!
        rqmc = setting.fixed_rqmc.rqmc(1:S,:);
end

C_lower     = diag(sqrt(Sig));
theta_all   = mu +  C_lower*rqmc';

end

function [clipped_gradLB_mu,clipped_gradLB_iSig] = Ng_clipping(gradLB_mu,gradLB_iSig,max_grad)

d_theta = numel(gradLB_mu);
Y12 = [gradLB_mu;gradLB_iSig];

if max_grad>0
    grad_norm = norm(Y12);
    norm_gradient_threshold = max_grad;
    if grad_norm > norm_gradient_threshold
        Y12 = (norm_gradient_threshold/grad_norm)*Y12;
    end
end

clipped_gradLB_mu = Y12(1:d_theta);
clipped_gradLB_iSig = Y12(d_theta+1:end);

end



