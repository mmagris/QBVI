function[y_hat,y,x,true_par] = sim_lr_data(N,b,u)
x = [ones(N,1),rand(N,2)];
b = b';
y_hat = x*b;
y =  y_hat+ normrnd(0,u,N,1);
true_par = [b;u];
end
