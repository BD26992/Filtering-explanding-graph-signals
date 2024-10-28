function [nmse_sequence,rnmse_sequence] = batch_eval(A_T,x_T,y_T,a_T,L,h)
%% Inputs
% A_T, x_T, a_T: adjacency, signal and attachment pattern stored sequentially
% L: filter order 
% gamma_batch: 2 norm Regularizer on batch cost.
%% Outputs
% nmse_sequence: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
%% Code
T=size(A_T,2)-1;l=1;
tic;
for i=1:T
A_ex=A_T{i};x_ex=x_T{i};att = a_T{i};
truth=y_T{i+1};%true ratings of current node stored in the last row of next x_T
%x_now=truth(end,:);items=find(x_now);
items=find(truth);% get the true ratings vector, then find the items rated a
xx=truth(items);%vector of ratings of new user after removing all zeros (we dont learn from them)
for j=1:length(items)     
A(l,:)=a_T{i}'*stack(A_T{i},x_ex(:,items(j)),L,0);% building the system matrix
x(l,1)=xx(j);
l=l+1;
end
end
tbatch=toc;
nmse_sequence=(((A*h-x).*(A*h-x))');
rnmse_sequence=sqrt(sum((A*h-x).*(A*h-x))/sum(x.*x));
rnmse_sequence=sqrt(mean(sum((A*h-x).*(A*h-x))))/(max(x)-min(x));
start=1;
%div=div_standard(A_T,x_T,a_T,start,10,L,h_batch);
end