function [squared_error,rnmse_seq,h,A_latest,x_latest,a_latest,y_latest] = online_pretrain(A_T,x_T,y_T,a_T,L,h_i)
%% Description
% Online learning for recsys type o data.
% Each incoming user has all data available at that time
% Everything is based on one user user graph
% All the ratings corresponding to one user are learnt together (sequentially)
% the final error plot is over ratings of all users.
%% Inputs
% A_T: Cell of adjacency matrices across time
% x_T: Cell of existing ratings matrix across time (x_T{i+1} is X_T{i} appended by one row)
% a_T: Cell of attachment patterns across time
% y_T: Cell containing Test ratings for users across time
% L: filter order
% step: learning rate
% mu: 2 norm reg weight for filter parameter
%% Outputs
% squared_error: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
% h: filter at end of online learning
%% Extract starting adj, signal
T=size(A_T,2)-1;p=1;
%% Initialize h
%h: filter coefficients; l: counter for all ratings; x_true: vector for
%holding all true values
tic;
l=1;x_true=[];
h=h_i;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T% Outer loop over users
 A_ex=A_T{i};x_ex=x_T{i};att = a_T{i};
%% Inner loop
truth=y_T{i+1};%true ratings of current node stored in the last row of next x_T
%x_now=truth(end,:);items=find(x_now);
items=find(truth);% get the true ratings vector, then find the items rated a
xx=truth(items);%vector of ratings of new user after removing all zeros (we dont learn from them)
for j=1:length(items)
%% Predict
A_x=stack(A_ex,x_ex(:,items(j)),L,0);% stacking with specific item
x_pred=att'*A_x*h;% prediction for this specific item
%% Error
   squared_error(l)=(x_pred-xx(j))^2;
l=l+1;%update counter
if mod(l,2000)==0
H(:,p)=h;p=p+1;
%grid=[-1:0.01:1]';
%vandermonde=fliplr(vander(grid));
%freq_res=vandermonde(:,1:L+1)*h;
%plot([-1:0.01:1],abs(freq_res));hold on;
end
end   
x_true=[x_true,xx];
end%updatevector of true values
%% plot frequency response of h
% if mod(l,100)==0
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_res=vandermonde(:,1:L+1)*h;
% plot([-1:0.01:1],abs(freq_res));hold on;
% end
tonline=toc;
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
%% Return variables
A_latest=A_T{end};
x_latest=x_T{end};
a_latest=a_T{end};
y_latest=y_T{end};
end
