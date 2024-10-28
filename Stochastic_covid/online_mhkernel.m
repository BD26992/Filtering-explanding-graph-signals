function [squared_error,rnmse_seq,theta,A_latest,x_latest,a_latest] = online_mhkernel(A_T,x_T,a_T,sigma,D,step,mu,hki,H)
%% Inputs
% A_T, x_T, a_T: adjacency, signal and attachment pattern stored sequentially
% sigma: kernel parameter
% D: dimensionality of kernel space parameter
% step: learning rate
% mu: 2 norm reg weight for filter parameter
%% Outputs
% squared_error: RNMSE of the whole sequence
%% Initialize
%theta=randn(2*D,1);
theta=randn(2*D,H);
theta=repmat(hki,1,H);
T=size(A_T,2)-1;
%% Online module
for i=1:T-1
 att = a_T{i}; next=x_T{i+1};x_true(i)=next(end);
 N=size(A_T{i},1);
%% Generate random feature for this attachment
A_next=[A_T{i},zeros(N,1);att',0];
Z=multihop_features(A_next,H,D,sigma);
%% Make prediction
x_pred=(1/H)*trace(theta'*Z);
%% Error
squared_error(i)=(x_pred-x_true(i))^2;
%% Update filter
%theta=theta-(step)*((x_pred-x_true(i))*z+2*mu*theta);
theta=theta-(step)*(2*(x_pred-x_true(i))*(1/H)*Z+2*mu*theta);
% if norm(theta)>=10000
%  theta=10000*theta/(norm(theta));
% end
end
%squared_error_2=sqrt(squared_error_2./norm(x_true,2)^2);
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
%% Return variables
A_latest=A_T{end};
x_latest=x_T{end};
a_latest=a_T{end};
end
