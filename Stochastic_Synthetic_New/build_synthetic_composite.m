function [A_Trn,x_Trn,a_Trn,D_Trn,A_Tst,x_Tst,a_Tst,D_Tst] =  build_synthetic_composite(nlinks,A_start,x_start,T,h,L,gen,frac)
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
w=median(A_start(find(A_start(:))));
%% Getting data-point at each time
A=cell(1,T);x=cell(1,T);a=cell(1,T);
A{1}=A_start;x{1}=x_start;type=0;N_t=N_start;D{1}=dictionary_maken(0.5*(A_start+A_start'));
gt=rand(5,1);gt(5)=0;
gt=project_onto_simplex(gt);
p=D{1}*gt;
for i=2:T
 [A{1,i},x{1,i},a{1,i-1}]=add_synthetic(A{i-1},x{i-1},p,w,h,nlinks,L,type,gen);% to make the time consistent
 N_t=N_t+1;% update number of nodes by one
 D{i}=dictionary_maken(0.5*(A{1,i}+A{1,i}'));
 p=D{i}*gt;
end
A_Trn=A(1,1:ceil(T*frac));A_Tst=A(1,ceil(T*frac)+1:end);
a_Trn=a(1,1:ceil(T*frac));a_Tst=a(1,ceil(T*frac)+1:end);
x_Trn=x(1,1:ceil(T*frac));x_Tst=x(1,ceil(T*frac)+1:end);
D_Trn=D(1,1:ceil(T*frac));D_Tst=D(1,ceil(T*frac)+1:end);
end