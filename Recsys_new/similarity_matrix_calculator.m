function [B,BI] = attachment_pattern(Y,X,NN)
%X: Represents N existing Node
%Y: Represents M incoming nodes
N=size(X,1);M=size(Y,1);O=zeros(size(X,1),size(Y,1));
for m=1:M
    for n=1:N
      a(n) = pearson(X(n,:),Y(m,:));
    end
[asort,indices]=sort(a);
att=zeros(N,1);
att(indices(1:NN),1)=a(indices(1:NN));
O(:,m)=att;
end
end