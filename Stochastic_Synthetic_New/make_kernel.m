function [K,x,alpha] =make_kernel(A,variance)
%% Inputs
% A: Adjacency matrix
% variance: kernel variance (See Shen et. al.)
%% Outputs
% K: Kernel matrix
% x: starting signal
%% Generate Kernel
N=size(A,1);K=zeros(N,N);
for i=1:N
    for j=1:N
        K(i,j)=exp(-norm(A(i,:)-A(j,:),2)^2/variance);
    end
end
%% Generate combining coefficients
alpha=0.5*ones(N,1)+0.5*rand(N,1);
alpha=randn(N,1);
x=K*alpha;
end