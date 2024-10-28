function [A_trn,x_trn,a_trn,y_trn,A_tst,x_tst,a_tst,y_tst,D_trn,D_tst]=build_recsys(X,Y,N_links,A_start,sim_type,frac)
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
A=cell(1,T);x=cell(1,T);a=cell(1,T);U=zeros(1,T);y=cell(1,T);D=cell(1,T);
A{1}=A_start;x{1}=X;% first cell of A an x contain the starting adj matrix and the starting ratings matrix
D{1}=dictionary_maken(0.5*(A_start+A_start'));
A_ex=A_start;
for i=2:T%this is to align the attachment vectors. for a fixed i, the attachemt at that cell i is related to the graph at i-1. Thus the true values will be at the next i
    [A{i},x{i},a{i-1},y{i-1}]=add_recsys_2(Y(i,:),x{i-1},N_links,A{i-1},sim_type);% attachment pattern
    N=size(A_ex,1);
    D{i}=dictionary_maken(0.5*(A{i}+A{i}'));
end
%% split 
A_trn=A(1,1:ceil(T*frac));A_tst=A(1,ceil(T*frac)+1:end);
x_trn=x(1,1:ceil(T*frac));x_tst=x(1,ceil(T*frac)+1:end);
a_trn=a(1,1:ceil(T*frac));a_tst=a(1,ceil(T*frac)+1:end);
y_trn=y(1,1:ceil(T*frac));y_tst=y(1,ceil(T*frac)+1:end);
D_trn=D(1,1:ceil(T*frac));D_tst=D(1,ceil(T*frac)+1:end);
end