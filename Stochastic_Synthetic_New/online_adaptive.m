function [squared_error2,rnmse_seq,h,A_latest,x_latest,a_latest,H,p_bar,w_bar,D,E] = online_adaptive(A_T,x_T,a_T,D_T,L,step,mu,nlinks,h_i,P,W,eta_p,eta_w,scale)
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
T=size(A_T,2)-1;l=1;
%% Initialize h
h=h_i;A_ex=A_T{1};N_s=size(A_ex,1);
D=D_T{1};E=(1/W)*rand(N_s,W);
p_bar=ones(P,1)/P;w_bar=ones(W,1)/W;
% figure;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T-1
 x_ex=x_T{i};
 N=size(A_ex,1);next=x_T{i+1};x_true(i)=next(end);%true signal taken from next instant (this is how data is organized)
%% Sample attachment from rule (To be used for stochatic setting)
 p=D*p_bar;
 w=E*w_bar;
%% Predict
A_x=stack(A_ex,x_ex,L,0);
x_pred(i)=scale*(w.*p)'*A_x*h;
%x_pred(i)=(att)'*A_x*h;

%% Error
   squared_error(i)=(x_pred(i)-x_true(i))^2;
   squared_error2(i)=squared_error(i)+mu*norm(h)^2;
%% Update filter
   h=h-step*(scale*(x_pred(i)-x_true(i))*A_x'*(w.*p)+scale^2*(A_x)'*diag(w.*w.*p.*(ones(N,1)-p))*(A_x)*h+2*mu*h);
   %h=h-step*((x_pred(i)-x_true(i))*A_x'*(att)+2*mu*h);
   %h=h-step*((A_x'*(att*att')*A_x+mu)*2*h-A_x'*att*x_true(i));
      % if norm(h)>=10
      %     h=10*h/norm(h);
      % end
%% Update attachment parameters p_bar, w_bar 
c=A_x*h;
grad_p=scale*(x_pred(i)-x_true(i))*D'*(w.*c)+scale^2*(D'*(c.*c.*w.*w)-D'*(p.*c.*c.*w.*w));
grad_w=scale*(x_pred(i)-x_true(i))*E'*(p.*c)+scale^2*2*E'*(w.*c.*c.*p.*(ones(N,1)-p));


%% alternate update
p_bar=p_bar-eta_p*grad_p;
p_bar=dual_projection(p_bar);
w_bar=w_bar-eta_w*grad_w;
w_bar=dual_projection(w_bar);

% p_bar=p_bar.*exp(-eta_p*grad_p)/(sum(p_bar.*exp(-eta_p*grad_p)));
% p_bar(find(p_bar<1e-6))=0;
% w_bar=w_bar.*exp(-eta_w*grad_w)/(sum(w_bar.*exp(-eta_w*grad_w)));

if length(find(isnan(p_bar))>0) || length(find(isnan(w_bar))>0)
squared_error2=[];
rnmse_seq=Inf;
A_latest=A_T{end};
x_latest=x_T{end};
a_latest=a_T{end};
H=[];
break;
end

%w_bar(find(w_bar<1e-6))=0;
%% Update A_ex
A_ex=[A_ex,zeros(N,1);a_T{i}',0];
D=D_T{i+1};E=[E;(1/W)*rand(1,W)];
%% plot frequency response of h
if mod(i,10)==0
grid=[-1:0.01:1]';
vandermonde=fliplr(vander(grid));
H(:,l)=abs(vandermonde(:,1:L+1)*h);
l=l+1;
end
end
 %squared_error_2=sqrt(squared_error_2./norm(x_true,2)^2);
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
A_latest=A_T{end};
x_latest=x_T{end};
a_latest=a_T{end};
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
end
