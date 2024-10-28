function [squared_error,rnmse_seq] = online_pretrain(A_T,x_T,a_T,L,h_i)
%% Inputs
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
T=size(A_T,2)-1;
%% Initialize h
h=h_i;
%h=rand(L+1,1);
%figure;
%% For element in sequence Get A{i}, x{i}, a{i}
for i=1:T-1
 A_ex=A_T{i};x_ex=x_T{i};att = a_T{i};
 N_ex=size(A_ex,1);next=x_T{i+1};x_true(i)=next(end);%true signal taken from next instant (this is how data is organized)
%% Sample attachment from rule (To be used for stochatic setting)

%% Predict
%A_x=stack(A_ex/abs(max(eig(A_ex))),x_ex,L,0);
A_x=stack(A_ex,x_ex,L,0);

x_pred=att'*A_x*h;
%% Error
   squared_error(i)=(x_pred-x_true(i))^2;
   squared_error_2(i)=(x_pred-x_true(i))^2;
%% Update filter
   %h=h-(step)*((x_pred-x_true(i))*[A_x'*att]+2*mu*h);
% if norm(h)>=10000
% h=10000*h/(norm(h));
% end
end
%% plot frequency response of h
% if mod(i,100)==0
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_res=vandermonde(:,1:L+1)*h;
% plot([-1:0.01:1],abs(freq_res));hold on;
% end
 %squared_error_2=sqrt(squared_error_2./norm(x_true,2)^2);
rnmse_seq=sqrt(sum(squared_error)/sum(x_true.*x_true));
rnmse_seq=sqrt(mean(squared_error))/(max(x_true)-min(x_true));
%% Return variables
end
