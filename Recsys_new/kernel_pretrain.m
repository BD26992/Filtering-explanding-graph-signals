function op = kernel_pretrain(A_T,x_T,frac,lambda,D,L,sigma)
%% Description
% pretraining the kernel parameter before online learning
%% Inputs
% A_T, x_T: as defined usually
% frac: fraction of existing nodes to train at
% existing: indices of existing nodes
% lambda: reg param.
%% Outputs
% op: learned param
%% Code
A_s=A_T{1};x_ex=x_T{1};%A_S, starting djacency matrix; y: target
N=size(A_s,1);
N_s=ceil(frac*N);T=[];
I=randperm(N_s);y=zeros(N_s,1);
for i=1:N_s
    a=zeros(1,D);b=zeros(1,D);
   for j=1:D
        v=sigma*randn(N,1);
        a(j)=sin(v'*A_s(I(i),:)');
        b(j)=cos(v'*A_s(I(i),:)');
   end
        t=(1/(sqrt(D)))*[a,b]';
        T=[T;t'];
        y(i,1)=x_ex(I(i),1);
end
op=inv(T'*T+lambda*eye(L+1))*T'*y;
end