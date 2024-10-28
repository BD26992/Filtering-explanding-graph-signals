function [A,x,a] =  add_synthetic(A_ex,x_ex,p,w,h,nlinks,L,type,gen)
%% Inputs
%A_ex,x_ex: Existing adjacency matrix and graph signal
%p,w: probability and weight vector
%nlinks: number of links formed
%type: type of stacking done (default 0)
%h, L: filter, filter order
%gen: way to generate data ('mean' or 'filter')
%% Outputs
%A,x: Appended adjacency matrix and graph signal
%a: attachment vector
%% Generate pattern
N=size(A_ex,1);
a = sample(p,w,N,nlinks);
A=[A_ex,zeros(N,1);a',0];
switch gen
case 'filter'
A_x=stack(A_ex,x_ex,L,type);
x_new=a'*A_x*h;
case 'mean'
b=a/sum(a);
x_new=x_ex'*b;
end
x=[x_ex;x_new];
%% Extension: call sample again for second type of graph, output the combined adjacency 
%and the second attachment
end