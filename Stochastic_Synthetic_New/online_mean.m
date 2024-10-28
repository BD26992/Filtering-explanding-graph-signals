function [rnmse_seq] = online_mean(A_T,x_T,x_0)
%% Inputs
% A_T: Cell of adjacency matrices for testing
% x_T: Cell of graph signals for testing
% a_T: Cell of attachment patterns for testing
% step: learning rate after hyperparameter selection
% lambda: 2 norm reg weight for filter parameter after hyperparameter selection
% h_i: filter from last stage of training with the hyperparam. combo
% A_0: Adj from last stage of training with the hyperparam. combo
% x_0: existing from last stage of training with the hyperparam. combo
% a_0: att from from last stage of training with the hyperparam. combo
%% Outputs
% squared_error: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
% h: last filter 
%% Extract starting adj, signal
T=size(A_T,2)-1;
%% For element in sequence Get A{i}, x{i}, a{i}
x_ex=x_0;
for i=1:T-1
next=x_T{i};x_true(i)=next(end);%true signal taken from next instant (this is how data is organized)
%% Sample attachment from rule (To be used for stochatic setting)

%% Predict
%A_x=stack(A_ex/abs(max(eig(A_ex))),x_ex,L,0);


x_pred=mean(x_ex);
%% Error
   squared_error(i)=(x_pred-x_true(i))^2;

x_ex=x_T{i};
end
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
end

