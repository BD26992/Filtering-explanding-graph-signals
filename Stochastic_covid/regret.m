function op = regret(x,c,y,u)
%% Description
% x is for an online algorithm
% y is normally for the batch
% c is regularizer weight for x
% u is the optimal filter
%% Code
% for a fair comparison
y=y+c*norm(u)^2*ones(1,length(y));
op=sum(x-y)/length(x);
end