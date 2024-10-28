function [min2,I] = min2d(A)
%% Description
% Find the minimum element in an array and return the corresponding index
%% Inputs
% A: 
%% Outputs
% min: minimum
% I: index (row, column)
%% Code
[M,Ir]=min(A);
[min2,Ic]=min(M);
I=[Ir(Ic),Ic];
end