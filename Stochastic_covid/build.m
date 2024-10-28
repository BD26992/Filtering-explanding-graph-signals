function [A_Trn,a_Trn,x_Trn,D_trn,A_Tst,a_Tst,x_Tst,D_tst] = build(R_e,R_i,T_b,T_eval,A_0,x_0,frac,N_links,C,hypsig,sim_type)
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
A=similarity([R_e(:,1:T_b);R_i(:,1:T_b)],[R_e(:,1:T_b);R_i(:,1:T_b)],sim_type,hypsig);%dense sim matrix
T_e=size(R_e,1);T=size(R_i,1);D=cell(1,T);
%declare cells for storing data
AT=cell(1,T);aT=cell(1,T);xT=cell(1,T);
existing=1:T_e;A_ex=A_0;N=size(A_0,1);
AT{1}=A_0;
D{1}=dictionary_maken(0.5*(A_0+A_0'));
mn=mean(x_0);
x_00=x_0-mn*ones(length(x_0),1);
xT{1}=x_00;
for i=2:T
  t=A(T_e+i-1,1:T_e+i-2);%similarity between current node and all existing nodes  
  [b,c]=sort(t,'descend');
  d=zeros(T_e+i-2,1);
  d(c(1:N_links))=b(1:N_links);
  aT{i-1}=d/C;
  xT{i}=[x_0;R_i(1:i-1,T_eval)];
  xT{i}=xT{i}-mn*ones(length(xT{i}),1);
  existing=[existing,T_e+i-1];%include the current node into existing
  AT{i}=[A_ex,zeros(N,1);d',0];
  A_ex=AT{i};
  N=N+1;
  %D{i}=dictionary_maken(0.5*(AT{i}+AT{i}'));
end
A_Trn=AT(1,1:ceil(T*frac));A_Tst=AT(1,ceil(T*frac)+1:end);
a_Trn=aT(1,1:ceil(T*frac));a_Tst=aT(1,ceil(T*frac)+1:end);
x_Trn=xT(1,1:ceil(T*frac));x_Tst=xT(1,ceil(T*frac)+1:end);
D_trn=D(1,1:ceil(T*frac));D_tst=D(1,ceil(T*frac)+1:end);
end
 