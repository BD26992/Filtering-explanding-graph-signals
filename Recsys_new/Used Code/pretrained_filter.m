function [h_batch]= pretrained_filter(A_ex,R,C,L,gamma_batch)
%% Description
% solves the pre-trained batch filter for starting users
%% Inputs
% A_ex: Starting adjacency matrix
% R: Ratings matrix for existing users
% C: number of ratings considered
% L: filter order
% gamma_batch: regularizer
%% Outputs
% h_batch: batch filter
[U,I]=size(R);S=R;c=0;
while c<=C
u=randperm(U);i=randperm(I);
if R(u(1),i(1))~=0
        c=c+1;
        x(c)=u(1);y(c)=i(1);z(c)=R(u(1),i(1));
        S(u(1),i(1))=0;
end
end
for i=1:length(x)
    item=y(i);
    item_graph_signal=S(:,item);
    Sh=stack(A_ex,item_graph_signal,L,1);
    B(i,:)=Sh(x(i),:);
end
h_batch=inv(B'*B+gamma_batch*eye(L+1))*B'*z';% least square batch solution
end