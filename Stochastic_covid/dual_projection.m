function x = dual_projection(y)
N=size(y,1);
lambda=randn(N,1);
mu=randn;
for n=1:100
lambda=mu*ones(N,1)-y;lambda(lambda<0)=0;
mu=(1/N)*(ones(1,N)*(y+lambda)-1);
end
x=y+lambda-mu*ones(N,1);
x(x<1e-4)=0;
end