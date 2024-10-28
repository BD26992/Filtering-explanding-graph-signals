function [squared_error,rnmse_seq,h] = online_pc_eval(A_T,x_T,a_T,L,step,mu,nlinks,h_i,A_0,x_0,a_0,scale)
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
T=size(A_T,2)-1;o=1;
%% Initialize h
h=h_i;
A_ex=A_0;x_ex=x_0;att=a_0;
weight=median(A_ex(find(A_ex(:))));
% figure;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T-1
 N=size(A_ex,1);next=x_T{i+1};x_true(i)=next(end);%true signal taken from next instant (this is how data is organized)
%% Sample attachment from rule (To be used for stochatic setting)
 p=ones(N,1)/N;
% p=sum(A_ex,2)/sum(sum(A_ex,2));
 w=weight*ones(N,1);
 
%% Predict
A_x=stack(A_ex,x_ex,L,0);
x_pred(i)=scale*(w.*p)'*A_x*h;
%x_pred(i)=(att)'*A_x*h;
%% Error
   squared_error(i)=(x_pred(i)-x_true(i))^2;
%% Update filter
   h=h-step*(scale*(x_pred(i)-x_true(i))*A_x'*(w.*p)+scale^2*(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x*h)+2*mu*h);
   h=h-step*((x_pred(i)-x_true(i))*A_x'*(att)+2*mu*h);
      if norm(h)>=10000
          h=10000*h/norm(h);
      end
%% Update A_ex
A_ex=[A_ex,zeros(N,1);att',0];
x_ex=x_T{i};att = a_T{i};
%% plot frequency response of h
% if mod(i,100)==0
% H(:,o)=h;o=o+1;
% end
end
 %squared_error_2=sqrt(squared_error_2./norm(x_true,2)^2);
 rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
A_latest=A_T{end};
x_latest=x_T{end};
a_latest=a_T{end};
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
end
