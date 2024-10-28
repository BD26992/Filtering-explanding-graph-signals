function time_varying_cumulative_regret = regret(x,c,y,u)
%% Description
% x is for an online algorithm
% y is normally for the batch
% c is regularizer weight for x
% u is the optimal filter
%% Code
% for a fair comparison
T=min(length(x),length(y));
y=y+c*norm(u)^2*ones(1,length(y));
time_vector=1:length(y);
time_varying_cumulative_regret=cumsum(x(1:length(y))-y)./time_vector;
end