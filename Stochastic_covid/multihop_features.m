function Z=multihop_features(A,H,D,sigma)
%% Description
% multi hop online kernel learning feature generation. 
%% Inputs:
% A: adjacency matrix which contains the attachment of the current incoming node
% H: number of multi hops considered
% D: feature specific constant
% sigma: variance of the kernel
%% Outputs
% Z: columnn i of Z contains the corresponding attachment vector for the i-th hop
%% Code
A=double(A>0);% A needs to be binarized (check the paper)
C=A;Z=zeros(2*D,H);
N=size(A,2);
for h=1:H
att=C(end,:)';
%% random feature
a=zeros(D,1);b=zeros(D,1);
for j=1:D
v=sigma*randn(N,1);
a(j)=sin(v'*att);
b(j)=cos(v'*att);
end
Z(:,h)=(1/(sqrt(D)))*[a;b];
C=A*C;
end
end