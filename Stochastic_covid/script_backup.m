clear all
%script for running simulations on covid forecasting
load seasurfacetemperature

%% Normalize the data matrix?
%R=normalize(R);

%% Split into existing and incoming
selection='random';N_start=50;
[R_e,R_i,existing,incoming]=select_users(R,selection,N_start);
%[R_e,R_i]=pre_process(Position,R);
%R_e=normalize(R_e,'column');R_i=normalize(R_i,'column');
%% Starting points
% Build G_0,A_0,x_0
T_b=1000;hypsig=1;
% time till which we will consider data to form all similarities. 
%These similarities will be used to build the existing graph as well as the connections of the incoming nodes.
T_e=1200; % time at which we consider the existing graph sigal
N_links=5;type='directed';sim_type='cosine';% self explanatory
[A_0,x_0]=starting_graph(R_e,T_b,T_e,N_links,type,sim_type,hypsig);
C=abs(max(eig(A_0)));
A_0=A_0/C;

%% Cleaning


%% Build data-set
H=0;% H isnthe forecasting window, i.e. equivalent to the time T_e+H
frac=0.8;% fraction of data used for the pre-trained filter
[A_Trn,a_Trn,x_Trn,A_Tst,a_Tst,x_Tst] = build(R_e,R_i,T_b,T_e+H,A_0,x_0,frac,N_links,C,hypsig,sim_type);


%% Batch
% gamma_batch=[1e-3,1e-2,1e-1,1,10];L=5;
% for i=1:length(gamma_batch)
% [se_sequence{i},rnmse_batch(i),h_batch{i}] = batch(A_Trn,x_Trn,a_Trn,L,gamma_batch(i));
% end
% rnmse_batch(isnan(rnmse_batch))=Inf;
% [min_batch,I_batch] = min(rnmse_batch);
% final_batch=batch_eval(A_Tst,x_Tst,a_Tst,L,h_batch{I_batch});
% 
% % batch frequency response
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_batch=abs(vandermonde(:,1:L+1)*h_batch{I_batch});



%% Online Learning
L=5;h_i=generative_filter(A_0,x_0,L,1,0.8);
% mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5];
% step_p=mu_p;
% for i=1:length(step_p)
%     for j=1:length(mu_p)
%          [squared_error_det{i,j},rnmse_det(i,j),h_det{i,j},A_latest_det{i,j},x_latest_det{i,j},a_latest_det{i,j},H_det{i,j}]=online_proposed(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),h_i);
%     end
% end
% rnmse_det(isnan(rnmse_det))=Inf;
% [min_det,I_det] = min2d(rnmse_det);
% 
% %Test Set evaluation
% [squared_error_det_test,rnmse_seq_det_test,h_det_test]=online_proposed_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_det(1)),mu_p(I_det(2)),h_det{I_det(1),I_det(2)},A_latest_det{I_det(1),I_det(2)},x_latest_det{I_det(1),I_det(2)},a_latest_det{I_det(1),I_det(2)});
% final_det=rnmse_seq_det_test;
% 
% 
% % Regret
% [se_sequence,~,h_batch] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_det(2)));
% regret_det_batch=regret(squared_error_det{I_det(1),I_det(2)},mu_p(I_det(2)),se_sequence,h_batch);


%% Online Stochastic Learning
% L=5;mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5];
% step_p=mu_p;rnmse_sto=zeros(length(step_p),length(mu_p));
% for i=1:length(step_p)
%     for j=1:length(mu_p)
%         rnmse_seq_sto=zeros(10,1);
%         for r=1:10
%             [squared_error_sto{r},rnmse_seq_sto(r),h_sto{i,j},A_latest_sto{i,j},x_latest_sto{i,j},a_latest_sto{i,j},~]=online_stochastic(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i);
%         end
%     rnmse_sto(i,j)=mean(rnmse_seq_sto);
%     end
% end
% rnmse_sto(isnan(rnmse_sto))=Inf;
% [min_sto,I_sto] = min2d(rnmse_sto);

% Test evaluation
% for r=1:200
% [~,~,h_sto,A_latest_sto,x_latest_sto,a_latest_sto,~]=online_stochastic(A_Trn,x_Trn,a_Trn,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_i);
% [~,rnmse_seq_sto_test(r),~]=online_stochastic_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_sto,A_latest_sto,x_latest_sto,a_latest_sto);
% end
% final_sto=mean(rnmse_seq_sto_test);
% 
% 
% % Regret
% [se_sequence,~,h_batch] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_sto(2)));
% for r=1:10
% [squared_error_sto,~,~,~,~,~,H_sto{r}]=online_stochastic(A_Trn,x_Trn,a_Trn,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_i);
% regret_sto_seq(r)=regret(squared_error_sto,mu_p(I_sto(2)),se_sequence,h_batch);
% end
% regret_sto_batch=mean(regret_sto_seq);
% freq_sto=avg_freq(H_sto);

%% Online pure stochastic Learning
% 
mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5];
step_p=mu_p;
for i=1:length(step_p)
    for j=1:length(mu_p)
       [squared_error_psto{i,j},rnmse_psto(i,j),h_psto{i,j},A_latest_psto{i,j},x_latest_psto{i,j},a_latest_psto{i,j}]=online_stochastic_pure(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i);
    end
end
rnmse_sto(isnan(rnmse_psto))=Inf;
[min_psto,I_psto] = min2d(rnmse_psto);

% Test
[squared_error_psto_test,rnmse_seq_psto_test,h_psto_test]=online_stochastic_pure_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_psto(1)),mu_p(I_psto(2)),N_links,h_psto{I_psto(1),I_psto(2)},A_latest_psto{I_psto(1),I_psto(2)},x_latest_psto{I_psto(1),I_psto(2)},a_latest_psto{I_psto(1),I_psto(2)});
final_psto=mean(rnmse_seq_psto_test);

% Regret
[se_sequence,~,h_batch] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_psto(2)));
regret_psto_batch=regret(squared_error_psto{I_psto(1),I_psto(2)},mu_p(I_psto(2)),se_sequence,h_batch);
 

%% Online adaptive
mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2];scale=0.1;
step_p=[1e-4,1e-3,1e-2,1e-1,1,10];P=10;W=10;
eta_p=1e-3;eta_w=1e-3;
rnmse_ada=zeros(length(step_p),length(mu_p));
for i=1:length(step_p)
for j=1:length(mu_p)
for r=1:10
[~,rnmse_seq_ada(r),~,~,~,~,~,p_t{i,j},w_t{i,j},~,~] = online_adaptive(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i,P,W,eta_p,eta_w,scale);
end
rnmse_ada(i,j)=mean(rnmse_seq_ada);
end
end
rnmse_ada(isnan(rnmse_ada))=Inf;
[min_ada,I_ada] = min2d(rnmse_ada);

% Test
for r=1:10
[~,~,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,~,p_bar,w_bar,D,E] = online_adaptive(A_Trn,x_Trn,a_Trn,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_i,P,W,eta_p,eta_w,scale);
[~,rnmse_seq_ada_test(r),~]=online_ada_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,P,W,eta_p,eta_w,p_bar,w_bar,D,E,scale);
end
final_ada=mean(rnmse_seq_ada_test);

% Regret
[se_sequence,~,h_batch] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_ada(2)));
for r=1:10
[squared_error_ada,~,~,~,~,~,H_ada{r},~,~,~,~]=online_adaptive(A_Trn,x_Trn,a_Trn,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_i,P,W,eta_p,eta_w,scale);
regret_ada_seq(r)=regret(squared_error_ada,mu_p(I_ada(2)),se_sequence,h_batch);
end
regret_ada_batch=mean(regret_ada_seq);
freq_ada=avg_freq(H_ada);
% 
%
 
%% Online Adaptive 2 (update only the probability of attachment)

mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2];
step_p=mu_p;P=10;W=10;
rnmse_ada2=zeros(length(step_p),length(mu_p));
eta_p=1e-3;eta_w=1e-3;
for i=1:length(step_p)
    for j=1:length(mu_p)
        for r=1:10
[~,rnmse_seq_ada2(r),~,~,~,~,~,p_bar2{i,j},~] = online_adaptive2(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i,P,eta_p);
        end
    rnmse_ada2(i,j)=mean(rnmse_seq_ada2);
    end
end
rnmse_ada2(isnan(rnmse_ada2))=Inf;
[min_ada2,I_ada2] = min2d(rnmse_ada2);

% Test set
for r=1:10
[~,~,h_ada2,A_latest_ada2,x_latest_ada2,a_latest_ada2,~,p_bar,D] = online_adaptive2(A_Trn,x_Trn,a_Trn,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_i,P,eta_p);
[~,rnmse_seq_ada_test2(r),~]=online_ada2_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_ada2,A_latest_ada2,x_latest_ada2,a_latest_ada2,P,eta_p,p_bar,D);
end
final_ada2=mean(rnmse_seq_ada_test2);


% Regret
[se_sequence,~,h_batch] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_ada2(2)));
for r=1:10
[squared_error_ada2,~,~,~,~,~,H_ada2{r},~,~]=online_adaptive2(A_Trn,x_Trn,a_Trn,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_i,P,eta_p);
regret_ada_seq2(r)=regret(squared_error_ada2,mu_p(I_ada2(2)),se_sequence,h_batch);
end
regret_ada2_batch=mean(regret_ada_seq2);
freq_ada2=avg_freq(H_ada2);


%% Online Kernel Learning
%sigma: kernel type, step: learning rate, mu: regularizer hyperparam, D: feature length
sigma=0.1;D=(L+1)/2;step=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
mu=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5];step=mu;
rnmse_kernel=zeros(length(step),length(mu));
hki = kernel_pretrain(A_Trn,x_Trn,0.5,0.1,D,L,sigma);
for i=1:length(step)
for j=1:length(mu)
[squared_error_ker,rnmse_seq_ker,h_ker{i,j},A_latest_ker{i,j},x_latest_ker{i,j},a_latest_ker{i,j}]=online_kernel(A_Trn,x_Trn,a_Trn,sigma,D,step(i),mu(j),hki);
rnmse_kernel(i,j)=mean(rnmse_seq_ker);
end
end
rnmse_kernel(isnan(rnmse_kernel))=Inf;
[min_ker,I_ker] = min2d(rnmse_kernel);

[squared_error_ker_test,rnmse_seq_ker_test,h_ker_test]=online_kernel_eval(A_Tst,x_Tst,a_Tst,sigma,D,step(I_ker(1)),mu(I_ker(2)),h_ker{I_ker(1),I_ker(2)},A_latest_ker{I_ker(1),I_ker(2)},x_latest_ker{I_ker(1),I_ker(2)},a_latest_ker{I_ker(1),I_ker(2)});

final_ker=rnmse_seq_ker_test;


%% Online multi-hop Kernel Learning

sigma=0.1;D=(L+1)/2;H=L;
step=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
mu=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5];
rnmse_mhkernel=zeros(length(step),length(mu));
hki = kernel_pretrain(A_Trn,x_Trn,0.5,0.1,D,L,sigma);
for i=1:length(step)
for j=1:length(mu)
[squared_error_mhker,rnmse_seq_mhker,h_mhker{i,j},A_latest_mhker{i,j},x_latest_mhker{i,j},a_latest_mhker{i,j}]=online_mhkernel(A_Trn,x_Trn,a_Trn,sigma,D,step(i),mu(j),hki,H);
rnmse_mhkernel(i,j)=mean(rnmse_seq_mhker);
end
end
rnmse_mhkernel(isnan(rnmse_mhkernel))=Inf;
[min_mhker,I_mhker] = min2d(rnmse_mhkernel);
[squared_error_mhker_test,rnmse_seq_mhker_test,h_mhker_test]=online_mhkernel_eval(A_Tst,x_Tst,a_Tst,sigma,D,H,step(I_mhker(1)),mu(I_mhker(2)),h_mhker{I_mhker(1),I_mhker(2)},A_latest_mhker{I_mhker(1),I_mhker(2)},x_latest_mhker{I_mhker(1),I_mhker(2)},a_latest_mhker{I_mhker(1),I_mhker(2)});
final_mhker=rnmse_seq_mhker_test;


%% Inductive transfer

gamma=[1e-3,1e-2,1e-1,1,10];
for i=1:length(gamma)
rnmse_pre_seq=zeros(10,1);ptfrac=0.5;
for r=1:10
[squared_error,rnmse_pre_seq(r)] = online_pretrain(A_Trn,x_Trn,a_Trn,L,h_i);
end
rnmse_pre(i)=mean(rnmse_pre_seq);
end
rnmse_pre(isnan(rnmse_pre))=Inf;
[min_pre,I_pre] = min(rnmse_pre);

%Test
for r=1:10
h=generative_filter(A_0,x_0,L,gamma(I_pre),0.5);    
[~,rnmse_seq_pre_test(r)]=online_pretrain(A_Tst,x_Tst,a_Tst,L,h);
end
final_pre=mean(rnmse_seq_pre_test);















