function [squared_error,rnmse_seq,h,p_bar] = online_ada2_eval(A_T,x_T,y_T,a_T,L,step,mu,nlinks,h_i,A_0,x_0,a_0,y_0,P,eta_p,p_i,D_i)
% A_T: Cell of adjacency matrices across time
% x_T: Cell of graph signals across time
% a_T: Cell of attachment patterns across time
% rule, nlinks: attachment rule, number of links
% L: filter
% step: learning rate
% lambda: 2 norm reg weight for filter parameter
%% Outputs
% squared_error: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
%% Extract starting adj, signal
T=size(x_T,2)-1;I=size(x_T{1},2);
%% Initialize h
%h: filter coefficients; l: counter for all ratings; x_true: vector for
%holding all true values
tic;
h=h_i;l=1;x_true=[];A_ex=A_0;x_ex=x_0;next_user_ratings=y_0;att=a_0;
weight=median(A_ex(find(A_ex(:))));
p_bar=p_i;D=D_i;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T% Outer loop over users
%% Inner loop
truth=next_user_ratings(end,:);%true ratings of current node stored in the last row of next x_T
%x_now=truth(end,:);items=find(x_now);
items=find(truth);% get the true ratings vector, then find the items rated a
xx=truth(items);%vector of ratings of new user after removing all zeros (we dont learn from them)
N=size(A_ex,1);
%% Diversity
%items_reccommended = [items_reccommended,div(h,x_ex,att,A_ex,L,10)];
for j=1:length(items)
%% Predict
p=D*p_bar;
w=weight*ones(N,1);
A_x=stack(A_ex,x_ex(:,items(j)),L,0);% stacking with specific item
x_pred=(w.*p)'*A_x*h;% prediction for this specific item
%% Error
   squared_error(l)=(x_pred-xx(j))^2;
%% Update filter
   %h=h-(step)*((x_pred-xx(j))*A_x'*att+2*mu*h);%update filter
   h=h-step*((x_pred-xx(j))*A_x'*(w.*p)+(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x)*h+2*mu*h);
   c=A_x*h;
grad_p=(x_pred-xx(j))*D'*(w.*c)+D'*(c.*c.*w.*w)-2*D'*(p.*c.*c.*w.*w);
p_bar=p_bar.*exp(-eta_p*grad_p)/(sum(p_bar.*exp(-eta_p*grad_p)));
p_bar(find(p_bar<1e-6))=0;
l=l+1;%update counter
end   
x_true=[x_true,xx];
%% Update A_ex
A_ex=A_T{i};x_ex=x_T{i};next_user_ratings=y_T{i};
D=[D;0.1*rand(1,P)];
end%updatevector of true values
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
%Div_score=length(unique(items_reccommended))/I;
end
