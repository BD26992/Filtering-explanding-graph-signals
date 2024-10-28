function [A_Trn,x_Trn,a_Trn,D_Trn,A_Tst,x_Tst,a_Tst,D_Tst] =  build_synthetic_kernel(rule,nlinks,A_start,T,variance,frac)
%% Inputs
%rule: attachment rule
%nlinks: number of links formed by each incoming node
%A_start: strating adjacency matrix
%x_start: starting graph signal
%T: length of sequence
%h, L: filter, filter order
%% Outputs
%OP: Structure containing all the relevant data
%% Setting the prob. from rule
N_start=size(A_start,1);
w=median(A_start(find(A_start(:))));D{1}=dictionary_maken(0.5*(A_start+A_start'));
% switch rule
%     case 'uniform'
%         p=ones(N_start,1)/N_start;
%     case 'degree-based'
%         p=sum(A_start,2)/sum(diag(A_start));
% end

%% Getting data-point at each time
A=cell(1,T);x=cell(1,T);a=cell(1,T);
A{1}=A_start;N_t=N_start;
gt=rand(5,1);gt(5)=0;
gt=project_onto_simplex(gt);
p=D{1}*gt;
for i=2:T
 [A{1,i},a{1,i-1}]=add_synthetic_kernel(A{i-1},p,w,nlinks);% to make the time consistent
 N_t=N_t+1;% update number of nodes by one
%  switch rule
%     case 'uniform'
%         p=ones(N_t,1)/N_t;
%     case 'degree-based'
%         p=sum(A{i},2)/sum(diag(A{i}));
%         %% This can also be performed by binarizing A{i}
% end
D{i}=dictionary_maken(0.5*(A{1,i}+A{1,i}'));
 p=D{i}*gt;
end
%% Now generate the signals on the final graph
[K,x_final,alpha] =make_kernel(A{T},variance);
x_final=x_final-mean(x_final)*ones(length(x_final),1);
for i=1:T
x{i}=x_final(1:N_start-1+i);
end
a{T}=[];A{T}=[];x{T}=[];% ignoring last sequence
%% This is extendable to second type of graph (DSLW) by including another p 
% within switch, and using the same sample function. add_synthetic will
%have one more outputs, another a. 

%% Output as structure
%OP = struct('adjacencies',A,'signals',x,'patterns',a);
A_Trn=A(1,1:ceil(T*frac));A_Tst=A(1,ceil(T*frac)+1:end);
a_Trn=a(1,1:ceil(T*frac));a_Tst=a(1,ceil(T*frac)+1:end);
x_Trn=x(1,1:ceil(T*frac));x_Tst=x(1,ceil(T*frac)+1:end);
D_Trn=D(1,1:ceil(T*frac));D_Tst=D(1,ceil(T*frac)+1:end);
end