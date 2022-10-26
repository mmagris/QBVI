function [h_func,llh,log_prior] = h_func_LinRegr_QBVI(data,theta,setting)

% Extract additional settings
d   = length(theta);
S0  = setting.Prior.Sig;
mu0 = setting.mu0;
Sig0_type = setting.Sig0_type;

% Extract data
y = data(:,1);
x = data(:,2:end);

% Compute log likelihood
b = theta(1:end-1);
s = exp(theta(end));

llh = -0.5*sum(log(2*pi*s^2)+(y-x*b).^2./s^2);


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