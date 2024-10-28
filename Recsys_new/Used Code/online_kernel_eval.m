function [squared_error,rnmse_seq] = online_kernel_eval(A_T,x_T,y_T,a_T,sigma,D,step,mu,theta_i,A_0,x_0,a_0,y_0)
%% Inputs
% A_T: Cell of adjacency matrices for testing
% x_T: Cell of graph signals for testing
% a_T: Cell of attachment patterns for testing
% step: learning rate after hyperparameter selection
% lambda: 2 norm reg weight for filter parameter after hyperparameter selection
% h_i: filter from last stage of training with the hyperparam. combo
% A_0: Adj from last stage of training with the hyperparam. combo
% x_0: existing from last stage of training with the hyperparam. combo
% a_0: att from from last stage of training with the hyperparam. combo
%% Outputs
% squared_error: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
% h: last filter 
%% Extract starting adj, signal
T=size(A_T,2)-1;
%% Initialize h
theta=theta_i;l=1;x_true=[];p=1;
%figure;
%% For element in sequence Get A{i}, x{i}, a{i}
A_ex=A_0;x_ex=x_0;att=a_0;truth=y_0;

for i=1:T% Outer loop over users
for j=1:D
v=sigma*randn(length(att),1);
a(j)=sin(v'*att);
b(j)=cos(v'*att);
end
z=(1/(sqrt(D)))*[a,b]';
%% Inner loop
%true ratings of current node stored in the last row of next x_T
%x_now=truth(end,:);items=find(x_now);
items=find(truth);% get the true ratings vector, then find the items rated a
xx=truth(items);%vector of ratings of new user after removing all zeros (we dont learn from them)
for j=1:length(items)
%% Predict
x_pred(l)=theta'*z;% prediction for this specific item
%% Error
   squared_error(l)=(x_pred(l)-xx(j))^2;
%% Update filter
   theta=theta-(step)*((x_pred(l)-xx(j))*z+2*mu*theta);
%   if norm(h)>1
%        h=h/norm(h);
%end
l=l+1;%update counter
end   
x_true=[x_true,xx];
truth=y_T{i}; A_ex=A_T{i};x_ex=x_T{i};att = a_T{i};
end
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
end