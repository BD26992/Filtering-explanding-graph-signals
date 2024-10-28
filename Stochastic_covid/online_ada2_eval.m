function [squared_error,rnmse_seq,h] = online_ada2_eval(A_T,x_T,a_T,L,step,mu,nlinks,h_i,A_0,x_0,a_0,P,eta_p,p_i,D_i)
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
A_ex=A_0;x_ex=x_0;
N_s=size(A_ex,1);
D=D_i;
weight=median(A_ex(find(A_ex(:))));
p_bar=p_i;
% figure;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T-1
 N=size(A_ex,1);next=x_T{i+1};x_true(i)=next(end);%true signal taken from next instant (this is how data is organized)
%% Sample attachment from rule (To be used for stochatic setting)
D=[D;0.1*rand(1,P)]; 
p=D*p_bar;
 w=weight*ones(N,1);
%% Predict
A_x=stack(A_ex,x_ex,L,0);
x_pred(i)=(w.*p)'*A_x*h;
%x_pred(i)=(att)'*A_x*h;
%% Error
   squared_error(i)=(x_pred(i)-x_true(i))^2;
%% Update filter
   h=h-step*((x_pred(i)-x_true(i))*A_x'*(w.*p)+(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x*h)+2*mu*h);
   %h=h-step*((x_pred(i)-x_true(i))*A_x'*(att)+2*mu*h);
   %h=h-step*((A_x'*(att*att')*A_x+mu)*2*h-A_x'*att*x_true(i));
%      if norm(h)>=10000
%          h=10000*h/norm(h);
%      end
%% Update attachment parameters p_bar, w_bar 
c=A_x*h;
grad_p=(x_pred(i)-x_true(i))*D'*(w.*c)+D'*(c.*c.*w.*w)-2*D'*(p.*c.*c.*w.*w);
p_bar=p_bar.*exp(-eta_p*grad_p)/(sum(p_bar.*exp(-eta_p*grad_p)));
p_bar(find(p_bar<1e-6))=0;
%% Update A_ex
A_ex=A_T{i};x_ex=x_T{i};att = a_T{i};
%% plot frequency response of h
% if mod(i,100)==0
% H(:,o)=h;o=o+1;
% end
end
 %squared_error_2=sqrt(squared_error_2./norm(x_true,2)^2);
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));

end
