function [A_0,x_0]=starting_graph(R_e,T_b,T_e,N_links,type,sim_type,hypsig)
%% Description
% construt the KNN directed graph of starting cities in covid data-set
%% Inputs
% R_e: existing data
% T_b: time horizon considered for similarity calculation
% T_e: existing time
% N_links: K in KNN
% type: directed (default)
%% Outputs (check paper for notation)
% A_0: starting adjacenncy matrix
% x_0: existing graph signal
%% Code
A=similarity(R_e(:,1:T_b),R_e(:,1:T_b),sim_type,hypsig);% dense A matrix
A_0 = NN(A,N_links,type);% make it KNN
x_0=R_e(:,T_e);
end