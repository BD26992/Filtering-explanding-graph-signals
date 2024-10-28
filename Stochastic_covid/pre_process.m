function [R_e,R_i]=pre_process(Position,R)
%% Description
% Divides the data into existing and "incoming" cities. First, we cluster
% based on position data. One cluster is assigned to the existing cities.
% Taking those indexes, we separate the R matrix into R_e and R_i. 
%% Inputs
% Position: latitude annd longitude of the 269 cities
% R: time varying data matrix of cities vs cases
%% Outputs
% R_e: existing data matrix
% R_i: incoming data matrix
%% Code
% Step 1: cluster positions (2 is a default choice)
[labels] = kmeans(Position,2);
if length(find(labels==1))>length(find(labels==2))
existing=find(labels==2);incoming=find(labels==1);
else
existing=find(labels==1);incoming=find(labels==2);
end
% Step 2: split R
R_e=R(existing,:);R_i=R(incoming,:);
end