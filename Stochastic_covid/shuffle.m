function op = shuffle(X)
%% Description
% shuffles rows on X
%% Code
N=size(X,1);
op=X(randperm(N),:);
end