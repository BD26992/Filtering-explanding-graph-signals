function [A_Trn,a_Trn,x_Trn,A_Tst,a_Tst,x_Tst] = build_pos(R_e,R_i,Position,existing,incoming,N_existing,N_incoming,T_b,T_eval,A_0,x_0,frac,N_links,C,hypsig,sim_type)
%% Description
% Build the training and testinng set for forecasting covid data at new cities.
% We start by building a dense similarity matrix.
% Then for each incoming node, we find its pattern, its signal, and update the matrices
%% Inputs
% R_e: existing data
% R_i: remaining data
% T_b: horizon for similarity calculation
% H: forecasting window (positive integer)
% A_0: starting adjacency matrix
% x_0: starting signal
% frac: percentage of incoming nodes in data set used for traininng
%% Outputs
% Trn: structure depicting training set
% Tst: structure depicting testing set
%% Code
D=squareform(pdist([Position(existing,:);Position(incoming,:)],'euclidean'));type='directed';
D_bar=sum(D(:))/(size(D,1)*(size(D,1)-1));
D=exp(-D/D_bar);
D=NN(D,N_links,type);
A=D(N_incoming+1:end,1:N_existing);
A=D;
T_e=size(R_e,1);T=size(R_i,1);
%declare cells for storing data
AT=cell(1,T);aT=cell(1,T);xT=cell(1,T);
existing=1:T_e;A_ex=A_0;N=size(A_0,1);
AT{1}=A_0;xT{1}=x_0;
for i=2:T
  t=A(T_e+i-1,1:T_e+i-2);%similarity between current node and all existing nodes  
  [b,c]=sort(t,'descend');
  d=zeros(T_e+i-2,1);
  d(c(1:N_links))=b(1:N_links);
  aT{i-1}=d/C;
  xT{i}=[x_0;R_i(1:i-1,T_eval)];
  existing=[existing,T_e+i-1];%include the current node into existing
  AT{i}=[A_ex,zeros(N,1);d',0];
  A_ex=AT{i};
  N=N+1;
end
A_Trn=AT(1,1:ceil(T*frac));A_Tst=AT(1,ceil(T*frac)+1:end);
a_Trn=aT(1,1:ceil(T*frac));a_Tst=aT(1,ceil(T*frac)+1:end);
x_Trn=xT(1,1:ceil(T*frac));x_Tst=xT(1,ceil(T*frac)+1:end);
end
 