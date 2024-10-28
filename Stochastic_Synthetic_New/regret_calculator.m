function op = regret_calculator(x,y)
%% Inputs
% x: error sequence from an algorithm
% y : batch error sequence
%% Outputs
% op: normalized regret over time
T=min(length(x),length(y));op=zeros(1,T);
for i=1:T
    op(i)=(sum(x(1:i))-sum(y(1:i)))/i;
end
end