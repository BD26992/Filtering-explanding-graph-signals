function h = generative_filter(A,x,L,gamma,frac)
%% Inputs
% A, x: Adjacency matrix, graph signal of starting graph, L: filter order
% gamma: reg hyperparameter
% frac=fraction of sampled nodes used to pre-train filter
%% Outputs
%h: trained filter
%% Generate Sampling matrix
N=length(x);t=zeros(N,1);index=randperm(N);t(index(1:ceil(frac*N)))=1;%generate sampling pattern
D=diag(t);
Ax=stack(A,D*x,L,1);%stack sampled graph signal only
h=pinv(Ax'*(eye(N)-D)*Ax+gamma*eye(L+1))*Ax'*(eye(N)-D)*x;%error at other vertices
end