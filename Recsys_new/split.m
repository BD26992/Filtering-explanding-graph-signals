function [f,g]=split(z,prob)
%% Description
% function that splits data from an online user into two parts, one for
% building connections to existing graph and other for testing
%% Inputs
% z: ratings vector (sparse) of current incoming users
% prob: probability with which each rating of this user is selcted for
% forming connections 
%% Outputs
% x: vector having those ratings which are used for forming connections
% y: vector having thode ratings used for prediction
%% Code
N=length(z);
a=find(z);
x=zeros(N,1);y=zeros(N,1);c=[];d=[];
f=zeros(1,N);g=zeros(1,N);
for i=1:length(a)
    b=rand;
    if b>(1-prob)
        c=[c,a(i)];
    else
        d=[d,a(i)];
    end
end
f(c)=z(c);g(d)=z(d);
end