function [h_func,llh,log_prior] = h_func_LogRegr_QBVI(data,theta,setting)

% Extract additional settings
d   = length(theta);
S0  = setting.Prior.Sig;
mu0 = setting.mu0;
Sig0_type = setting.Sig0_type;

% Extract data
y = data(:,1);
x = data(:,2:end);

% Compute log likelihood
w = theta(2:end);
b = theta(1);

aux = x*w+b;
llh = y.*aux-log(1+exp(aux));
llh = sum(llh);


% Compute log prior
aux = (theta-mu0);

if Sig0_type == 1
    log_prior = -d/2*log(2*pi)-d/2*sum(log(S0))-1/2*aux'*(aux./S0); % scalar precision
else 
    log_prior = -d/2*log(2*pi)-1/2*sum(log(S0))-1/2*aux'*(aux./S0); % vector
end

% Compute h(theta) = log p(y|theta) + log p(theta)
h_func = llh + log_prior;

end