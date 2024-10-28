function [x]=build_recsys_stochastic(X,Y)
%% Description
% Builds incoming node data-set 
%% Inputs
% X: U_existing X I ratings matrix
% Y: U_online X I ratings matrix
% N_links: number of connections formed
% A_start: starting similarity matrix
% sim_type: similarity metric
%% Outputs
% A: cell containing adjacency matrices
% x: cell containing signals
% a: cell containing attachment patterns
%% Code
T=size(Y,1);
x=cell(1,T);
x{1}=X;% first cell of A an x contain the starting adj matrix and the starting ratings matrix
for i=2:T%this is to align the attachment vectors. for a fixed i, the attachemt at that cell i is related to the graph at i-1. Thus the true values will be at the next i
    [x{i}]=add_recsys_stochastic(Y(i,:),x{i-1});% attachment pattern
end
end