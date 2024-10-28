function [A]=normalize(R_e)
%% Description: normalization done on R_e
%% Inputs
%% Outputs
%% Code
[N,M]=size(R_e);
for i=1:N
mx=mean(R_e(i,:));sdx=std(R_e(i,:));
A(i,:)=(R_e(i,:)-mx*ones(1,M))/sdx;
end
end