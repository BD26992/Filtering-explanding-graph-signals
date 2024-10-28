function [x]=add_recsys_stochastic(z,Y)
%% Description
% function that calculates similarity between z and rows of Y (ratings)
%% Inputs
% z: ratings vector (sparse) of current incoming users
% Y: ratings matrix of existing users (rating vectors stored row-wise)
% N_links: number of links needed
%% Outputs
% A: updated adj matrix
% x: updated ratings matrix (simple row append)
% att: attachment pattern
%% Code
N_ex=size(Y,1);att=zeros(N_ex,1);
%% New part
%[m,n]=split(z,0.5);%split z, all the ratings of the current incoming user into m(for building links) and n(for online prediction), each element of m selected with prob 0.5

%a=similarity(m,Y,sim_type);%vector of similarities between new user and existing ones, based on m
%[c,b]=sort(a,'descend');
%att(b(1:N_links))=c(1:N_links);%attach pattern being built based on m and existing ratings 

%A=[A_ex,zeros(N_ex,1);att',0];%new adjacency matrix
x=[Y;z];%new ratings matrix, by incorporating m
end
