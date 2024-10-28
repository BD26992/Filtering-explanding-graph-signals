clear all

%% load and remove low interactions
load R
item_threshold=10;user_threshold=10;
[R]=clean(ratings,user_threshold,item_threshold);

%% Split along items and users to get Block matrices 
[U,I]=size(R);
nia=600;
nexu=300;
t1=randperm(I);item_attachment_indices=t1(1:nia);item_training=t1(nia+1:end);
t2=randperm(U);user_ex=t2(1:nexu);user_in=t2(nexu+1:end);
B1=R(user_ex,item_attachment_indices);%B1: Block 1
B2=R(user_ex,item_training);%B2: Block 2
B3=R(user_in,item_attachment_indices);%B3: Block 3
B4=R(user_in,item_training);%B4: Block 4

%% Focus on one item from training split for the experiment
select=8;
item_ID=item_training(select);
x_0=B2(:,select);
x_in=B4(:,select);

%% Build the starting Adjacency Matrix
NN=15;
A_0 = attachment_pattern(B1,B1,NN);
%% Generating the attachment vectors
[valid,~]=find(x_in);
%Truncate B3 as we don't need those users who have zero ratings
B3_eff=B3(valid,:);
%Similarly truncate the ratings vector
x_in=x_in(valid);
Total_incoming=size(B3_eff,1);
B1_copy=B1;
a_t=cell(Total_incoming,1);x_t=cell(Total_incoming,1);A_t=cell(Total_incoming,1);A_ex=A_0;
for i=1:Total_incoming
    a_t{i}=attachment_pattern(B3_eff(i,:),B1_copy,NN);
    B1_copy=[B1_copy;B3_eff(i,:)];
    x_t{i}=x_in(i);
    A_t{i}=[A_ex,zeros(length(a_t{i}),1);a_t{i},0];
    A_ex=A_t{i};
end

%% Separate into training set cell and test set cell
% Separate x_t, A_t,a_t
splitting_fraction=0.7;
N_trn=ceil(splitting_fraction*Total_incoming);
A_trn=A_t(1:N_trn);a_trn=a_t(1:N_trn);x_trn=x_t(1:N_trn);
A_tst=A_t(N_trn+1:end);a_tst=a_t(N_trn+1:end);x_tst=x_t(N_trn+1:end);


%% Training and evaluation for each method

%% Online Learning
L=5;h_i=generative_filter(A_0,x_0,L,1,0.8);
mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5];
step_p=mu_p;
for i=1:length(step_p)
    for j=1:length(mu_p)
         [squared_error_det{i,j},rnmse_det(i,j),h_det{i,j},A_latest_det{i,j},x_latest_det{i,j},a_latest_det{i,j},H_det{i,j}]=online_proposed(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),h_i);
    end
end
rnmse_det(isnan(rnmse_det))=Inf;
[min_det,I_det] = min2d(rnmse_det);










