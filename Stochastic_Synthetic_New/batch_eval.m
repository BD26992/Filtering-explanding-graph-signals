function [rnmse_sequence] = batch_eval(A_T,x_T,a_T,L,h_batch)
%% Inputs
% A_T, x_T, a_T: adjacency, signal and attachment pattern stored sequentially
% L: filter order 
% gamma_batch: 2 norm Regularizer on batch cost.
%% Outputs
% nmse_sequence: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
%% Code
T=size(A_T,2)-1;
for i=1:T-1
A(i,:)=a_T{i}'*stack(A_T{i},x_T{i},L,0);% building the system matrix
t=x_T{i+1};x(i,1)=t(end);
end
%nmse_sequence=(((A*h_batch-x).*(A*h_batch-x))');
rnmse_sequence=sqrt(sum((A*h_batch-x).*(A*h_batch-x))/sum(x.*x));
rnmse_sequence=sqrt(mean(sum((A*h_batch-x).*(A*h_batch-x))))/(max(x)-min(x));
end