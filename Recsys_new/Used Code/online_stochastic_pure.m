function [squared_error2,rnmse_seq,h,A_latest,x_latest,a_latest,y_latest,H] = online_stochastic_pure(A_start,x_T,y_T,a_T,L,step,mu,nlinks,h_i)
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
T=size(x_T,2)-1;o=1;items_reccommended=[];I=size(x_T{1},2);
%% Initialize h
%h: filter coefficients; l: counter for all ratings; x_true: vector for
%holding all true values
tic;
h=h_i;l=1;x_true=[];A_ex=A_start;weight=median(A_ex(find(A_ex(:))));
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T% Outer loop over users
 x_ex=x_T{i};next_user_ratings=y_T{i};
%% Inner loop
truth=next_user_ratings(end,:);%true ratings of current node stored in the last row of next x_T
%x_now=truth(end,:);items=find(x_now);
items=find(truth);% get the true ratings vector, then find the items rated a
xx=truth(items);%vector of ratings of new user after removing all zeros (we dont learn from them)
N=size(A_ex,1);p=sum(A_ex,2)/sum(sum(A_ex,2));
w=weight*ones(N,1);
att=a_T{i};
%% Diversity
%items_reccommended = [items_reccommended,div(h,x_ex,att,A_ex,L,10)];
for j=1:length(items)
%% Predict
A_x=stack(A_ex,x_ex(:,items(j)),L,0);% stacking with specific item
x_pred=(w.*p)'*A_x*h;
%% Error
   squared_error(l)=(x_pred-xx(j))^2;
   squared_error2(l)=((w.*p)'*A_x*h-xx(j))^2+(A_x*h)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x*h)+mu*norm(h)^2;
%% Update filter
   h=h-step*((x_pred-xx(j))*A_x'*(w.*p)+(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x*h)+2*mu*h);
   h=h-step*((x_pred-xx(j))*A_x'*(att)+2*mu*h);
%   if norm(h)>1
%        h=h/norm(h);
%end
l=l+1;%update counter
if mod(l,10)==0
grid=[-1:0.01:1]';
vandermonde=fliplr(vander(grid));
H(:,o)=abs(vandermonde(:,1:L+1)*h);
o=o+1;
end
end   
x_true=[x_true,xx];
%% Update A_ex
A_ex=[A_ex,zeros(N,1);a_T{i}',0];
end%updatevector of true values
%% plot frequency response of h
% if mod(l,100)==0
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_res=vandermonde(:,1:L+1)*h;
% plot([-1:0.01:1],abs(freq_res));hold on;
% end
tonline=toc;
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
%Div_score=length(unique(items_reccommended))/I;
A_latest=A_ex;
x_latest=x_T{end};
a_latest=a_T{end};
y_latest=y_T{end};
end
