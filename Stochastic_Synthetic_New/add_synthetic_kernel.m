function [A,a] =  add_synthetic_kernel(A_ex,p,w,nlinks)
%% Inputs
%A_ex,x_ex: Existing adjacency matrix and graph signal
%p,w: probability and weight vector
%nlinks: number of links formed
%type: type of stacking done (default 0)
%h, L: filter, filter order
%% Outputs
%A,x: Appended adjacency matrix and graph signal
%a: attachment vector
%% Generate pattern
N=size(A_ex,1);
a = sample(p,w,N,nlinks);
A=[A_ex,zeros(N,1);a',0];
%% Extension: call sample again for second type of graph, output the combined adjacency 
%and the second attachment
end