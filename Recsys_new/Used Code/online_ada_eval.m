function [squared_error,rnmse_seq,h,p_bar,w_bar] = online_ada_eval(A_T,x_T,y_T,a_T,D_T,L,step,mu,nlinks,h_i,A_0,x_0,a_0,y_0,P,W,eta_p,eta_w,p_i,w_i,D_i,E_i,C)
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
T=size(x_T,2)-1;o=1;I=size(x_T{1},2);
%% Initialize h
%h: filter coefficients; l: counter for all ratings; x_true: vector for
%holding all true values
tic;
h=h_i;l=1;x_true=[];A_ex=A_0;x_ex=x_0;next_user_ratings=y_0;att=a_0;
weight=median(A_ex(find(A_ex(:))));
p_bar=p_i;w_bar=w_i;D=D_i;E=E_i;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T% Outer loop over users
%% Inner loop
truth=next_user_ratings(end,:);%true ratings of current node stored in the last row of next x_T
%x_now=truth(end,:);items=find(x_now);
items=find(truth);% get the true ratings vector, then find the items rated a
xx=truth(items);%vector of ratings of new user after removing all zeros (we dont learn from them)
N=size(A_ex,1);
%D=rand(N,P);E=0.1*rand(N,W);
%p_bar=ones(P,1)/P;w_bar=ones(W,1)/W;
%% Diversity
%items_reccommended = [items_reccommended,div(h,x_ex,att,A_ex,L,10)];
for j=1:length(items)
%% Predict
p=D*p_bar;
w=E*w_bar;
%att=sample(p,w,N,nlinks);
A_x=stack(A_ex,x_ex(:,items(j)),L,0);% stacking with specific item
%x_pred=att'*A_x*h;% prediction for this specific item
x_pred=(w.*p)'*A_x*h;
%% Error
   squared_error(l)=(x_pred-xx(j))^2;
%% Update filter
   %h=h-(step)*((x_pred-xx(j))*A_x'*att+2*mu*h);%update filter
   h=h-step*((x_pred-xx(j))*A_x'*(w.*p)+(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x)*h+2*mu*h);
   c=A_x*h;
grad_p=(x_pred-xx(j))*D'*(w.*c)+D'*(c.*c.*w.*w)-2*D'*(p.*c.*c.*w.*w)+p_bar;
grad_w=(x_pred-xx(j))*E'*(p.*c)+2*E'*(w.*c.*c.*p.*(ones(N,1)-p));
%% alternate update
p_bar=p_bar-eta_p*grad_p;
p_bar=dual_projection(p_bar);
w_bar=w_bar-eta_w*grad_w;
w_bar=dual_projection(w_bar);
%   if norm(h)>1
%        h=h/norm(h);
%end
l=l+1;%update counter
if mod(l,2000)==0
H(:,o)=h;o=o+1;
%grid=[-1:0.01:1]';
%vandermonde=fliplr(vander(grid));
%freq_res=vandermonde(:,1:L+1)*h;
%plot([-1:0.01:1],abs(freq_res));hold on;
end
end   
x_true=[x_true,xx];
%% Update A_ex
A_ex=A_T{i};x_ex=x_T{i};next_user_ratings=y_T{i};att=a_T{i};
E=[E;C*rand(1,W)];D=D_T{i};
end%updatevector of true values
%% plot frequency response of h
% if mod(l,100)==0
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_res=vandermonde(:,1:L+1)*h;
% plot([-1:0.01:1],abs(freq_res));hold on;
% end
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
%Div_score=length(unique(items_reccommended))/I;
end
