function [squared_error,rnmse_seq,h] = online_ada_eval(A_T,x_T,a_T,D_T,L,step,mu,nlinks,h_i,A_0,x_0,a_0,P,W,eta_p,eta_w,p_i,w_i,D_i,E_i,scale)
%% Description
% Online adaptive filtering without knwing the attachments at any time
%% Inputs
% A_T: Cell of adjacency matrices across time
% x_T: Cell of graph signals across time
% L: filter
% step: learning rate
% mu: 2 norm reg weight for filter parameter
% rule, nlinks: attachment rule, number of links
%% Outputs
% squared_error: Squared error across sequence
% rnmse_seq: RNMSE of the entire sequence
%% Extract starting adj, signal
T=size(A_T,2)-1;
%% Initialize h
h=h_i;
A_ex=A_0;x_ex=x_0;att=a_0;
D=D_i;E=[E_i;rand(1,W)];
p_bar=p_i;w_bar=w_i;
% figure;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T-1
 N=size(A_ex,1);next=x_T{i+1};x_true(i)=next(end);%true signal taken from next instant (this is how data is organized)
%% Sample attachment from rule (To be used for stochatic setting)

p=D*p_bar;
w=E*w_bar;
%att=sample(p,w,N,nlinks);
%% Predict
A_x=stack(A_ex,x_ex,L,0);
x_pred(i)=(w.*p)'*A_x*h;
%x_pred(i)=(att)'*A_x*h;
%% Error
   squared_error(i)=(x_pred(i)-x_true(i))^2;
%% Update filter
   h=h-step*((x_pred(i)-x_true(i))*A_x'*(w.*p)+(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x*h)+2*mu*h);
%% Update attachment parameters p_bar, w_bar 
c=A_x*h;
grad_p=(x_pred(i)-x_true(i))*D'*(w.*c)+D'*(c.*c.*w.*w)-2*D'*(p.*c.*c.*w.*w);
grad_w=(x_pred(i)-x_true(i))*E'*(p.*c)+2*E'*(w.*c.*c.*p.*(ones(N,1)-p));
p_bar=p_bar-eta_p*grad_p;
p_bar=dual_projection(p_bar);
w_bar=w_bar-eta_w*grad_w;
w_bar=dual_projection(w_bar);
%% Update A_ex
A_ex=A_T{i};x_ex=x_T{i};att = a_T{i};E=[E;rand(1,W)];D=D_T{i};
%% plot frequency response of h
% if mod(i,100)==0
% H(:,o)=h;o=o+1;
% end
end
 %squared_error_2=sqrt(squared_error_2./norm(x_true,2)^2);
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));

end
