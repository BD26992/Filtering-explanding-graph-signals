clear all
%script for generating recsys type data and then performing online learning
%over it
%% Obtain Data matrix
load R;

%% Clean data
item_threshold=20;user_threshold=20;
[R]=clean(ratings,user_threshold,item_threshold);

%% Preprocessing
%selection: mode of selecting existing users. Options 1)'random' 2)'most rated'
selection='max ratings';N_start=500;
[U_e,U_o]=select_users(R,selection,N_start);
non_zero_elements = U_e(U_e ~= 0);
mean_val = mean(non_zero_elements);
std_val = std(non_zero_elements);
U_e=normalize_recsys(U_e,mean_val,std_val);
U_o=normalize_recsys(U_o,mean_val,std_val);
for z=1:1
U_o=shuffle(U_o);
%% Normalize data (?)
%% Similarity matrix between existing users
sim_type='cosine';N_links=31;L=15;
A_start=similarity(U_e,U_e,sim_type);

%% Sparsify A_start
A_start=NN(A_start,N_links,'directed');frac=0.7;
A_start=A_start/abs(max(eig(A_start)));
A_start(A_start<0)=0;
%A_start=inv(diag((sum(A_start,2))))*A_start;
%% Generate data-set
[A_Trn,x_Trn,a_Trn,y_Trn,A_Tst,x_Tst,a_Tst,y_Tst,D_trn,D_tst]=build_recsys(U_e,U_o,N_links,A_start,sim_type,frac);


%% Batch

% gamma_batch=[1e-3,1e-2,1e-1,1,10];
% for i=1:length(gamma_batch)
% [~,rnmse_batch(i),h_batch{i}] = deterministic_batch(A_Trn,x_Trn,y_Trn,a_Trn,L,gamma_batch(i));
% end
% rnmse_batch(isnan(rnmse_batch))=Inf;
% [min_batch,I_batch] = min(rnmse_batch);
% [squared_error_batch{z},final_batch(z)]=batch_eval(A_Tst,x_Tst,y_Tst,a_Tst,L,h_batch{I_batch});
% median_batch(z)=median(squared_error_batch{z});
% 
% % batch frequency response
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_batch(:,z)=abs(vandermonde(:,1:L+1)*h_batch{I_batch});
% % 
% 
% 
% 
%% Online Learning 
% mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
% step_p=[1e-6,1e-5,1e-4,1e-3,1e-2];
C=10000;gamma_batch=1;
h_i= pretrained_filter(A_start,U_e,C,L,gamma_batch);
% Descriptions
% for i=1:length(step_p)
%     for j=1:length(mu_p)
%          [squared_error_det{i,j},rnmse_det(i,j),h_det{i,j},A_latest_det{i,j},x_latest_det{i,j},a_latest_det{i,j},y_latest_det{i,j},H_det{i,j}]=online_proposed(A_Trn,x_Trn,a_Trn,y_Trn,L,step_p(i),mu_p(j),h_i);
%     end
% end
% rnmse_det(isnan(rnmse_det))=Inf;
% [min_det,I_det] = min2d(rnmse_det);


% Test 

% [squared_error_det_test{z},rnmse_seq_det_test,h_det_test]=online_proposed_eval(A_Tst,x_Tst,a_Tst,y_Tst,L,step_p(I_det(1)),mu_p(I_det(2)),h_det{I_det(1),I_det(2)},A_latest_det{I_det(1),I_det(2)},x_latest_det{I_det(1),I_det(2)},a_latest_det{I_det(1),I_det(2)},y_latest_det{I_det(1),I_det(2)});
% final_det(z)=rnmse_seq_det_test;
% median_det(z)=median(squared_error_det_test{z});
% % Regret
% [se_sequence,~,hb] = deterministic_batch(A_Trn,x_Trn,y_Trn,a_Trn,L,mu_p(I_det(2)));
% regret_det_batch(z)=regret(squared_error_det{I_det(1),I_det(2)},mu_p(I_det(2)),se_sequence,hb);
% freq_det{z}=H_det{I_det(1),I_det(2)};

%% Online Stochastic Learning
% 
% mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
% step_p=[1e-6,1e-5,1e-4,1e-3,1e-2];
% for i=1:length(step_p)
%     for j=1:length(mu_p)
%         for r=1:10
%             [~,rnmse_seq_sto(r),~,~,~,~,~,~]=online_stochastic(A_start,x_Trn,y_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i);
%         end
%     rnmse_sto(i,j)=mean(rnmse_seq_sto);
%     end
% end
% rnmse_sto(isnan(rnmse_sto))=Inf;
% [min_sto,I_sto] = min2d(rnmse_sto);
% 
% %Test
% for r=1:1
% [~,~,h_sto,A_latest_sto,x_latest_sto,a_latest_sto,y_latest_sto,~]=online_stochastic(A_start,x_Trn,y_Trn,a_Trn,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_i);
% [~,rnmse_seq_sto_test(r),~]=online_stochastic_eval(A_Tst,x_Tst,a_Tst,y_Tst,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_sto,A_latest_sto,x_latest_sto,a_latest_sto,y_latest_sto);
% end
% final_sto(z)=mean(rnmse_seq_sto_test);
% 
% 
% final_mean(z) = online_mean(A_Tst,x_Tst,y_Tst,x_Trn{end},y_Trn{end});
%Regret
% [se_sequence,~,hb] = deterministic_batch(A_Trn,x_Trn,y_Trn,a_Trn,L,mu_p(I_sto(2)));
% for r=1:10
% [squared_error_sto,~,~,~,~,~,~,H_sto{r}]=online_stochastic(A_start,x_Trn,y_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i);
% regret_sto_seq(r)=regret(squared_error_sto,mu_p(I_sto(2)),se_sequence,hb);
% end
% regret_sto_batch(z)=mean(regret_sto_seq);
% freq_sto{z}=avg_freq(H_sto);


%% Online pure stochastic Learning
% 

%h_i=generative_filter(A_start,x_0,L,1,0.5);

 mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
 step_p=[1e-6,1e-5,1e-4,1e-3,1e-2];
 for i=1:length(step_p)
     for j=1:length(mu_p)
     [squared_error_stop{i,j},rnmse_stop(i,j),h_stop{i,j},A_latest_stop{i,j},x_latest_stop{i,j},a_latest_stop{i,j},y_latest_stop{i,j},H_psto{i,j}]=online_stochastic_pure(A_start,x_Trn,y_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i);
%     %regret_stop_det(i,j)=sum(squared_error_det{i,j}-squared_error_stop{i,j})/size(x_Trn,2);
     end
 end
 rnmse_stop(isnan(rnmse_stop))=Inf;
 [min_stop,I_stop] = min2d(rnmse_stop);
% %regret_stop_det=sum(squared_error_det{i,j}-squared_error_stop{i,j})/size(x_Trn,2);
% %Test
 % [squared_error_stop_test{z},rnmse_seq_psto_test,h_stop_test]=online_stochastic_pure_eval(A_Tst,x_Tst,a_Tst,y_Tst,L,step_p(I_stop(1)),mu_p(I_stop(2)),N_links,h_stop{I_stop(1),I_stop(2)},A_latest_stop{I_stop(1),I_stop(2)},x_latest_stop{I_stop(1),I_stop(2)},a_latest_stop{I_stop(1),I_stop(2)},y_latest_stop{I_stop(1),I_stop(2)});
 % final_stop(z)=mean(rnmse_seq_psto_test);
 % median_stop(z)=median(squared_error_stop_test{z});


% % 
% % 
% % %Regret
[se_sequence,~,hb] = deterministic_batch(A_Trn,x_Trn,y_Trn,a_Trn,L,mu_p(I_stop(2)));
[regret_psto_batch{z}]=regret(squared_error_stop{I_stop(1),I_stop(2)},mu_p(I_stop(2)),se_sequence,hb);
plot(regret_psto_batch{z});hold on;
% freq_psto{z}=H_psto{I_stop(1),I_stop(2)};
% % 
%% Online adaptive
%

mu_p=[1e-6,1e-3,1e-1];
step_p=[1e-6,1e-3,1e-1];P=5;W=1;
rnmse_ada=zeros(length(step_p),length(mu_p));
% %h_i= pretrained_filter(A_start,U_e,10000,L,1);
eta_p=1;eta_w=1;C=10000;
Constant=1e-2;
for i=1:length(step_p)
     for j=1:length(mu_p)
         for r=1:1
 [squared_error_ada2{i,j},rnmse_seq_ada(r),~,~,~,~,~,~,p_t{i,j},w_t{i,j},~,~] = online_adaptive(A_start,x_Trn,y_Trn,a_Trn,D_trn,L,step_p(i),mu_p(j),N_links,h_i,P,W,eta_p,eta_w,Constant);
         end
     rnmse_ada(i,j)=mean(rnmse_seq_ada);
     %regret_ada_det(i,j)=sum(squared_error_det{i,j}-squared_error_ada{i,j})/size(x_Trn,2);
     end
end
rnmse_ada(isnan(rnmse_ada))=Inf;
[min_ada,I_ada] = min2d(rnmse_ada);
% 
% % Test
% for r=1:10
% [~,~,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,y_latest_ada,~,p_bar,w_bar,D,E] = online_adaptive(A_start,x_Trn,y_Trn,a_Trn,D_trn,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_i,P,W,eta_p,eta_w,Constant);   
% [squared_error_ada_test{r},rnmse_seq_ada_test(r),~,p_f,w_f]=online_ada_eval(A_Tst,x_Tst,y_Tst,a_Tst,D_tst,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,y_latest_ada,P,W,eta_p,eta_w,p_bar,w_bar,D,E,Constant);
% en(r)=median(squared_error_ada_test{r});
% end
% final_ada(z)=mean(rnmse_seq_ada_test);
% 
% median_ada(z)=mean(en);
% 
% % Regret
[se_sequence,~,hb] = deterministic_batch(A_Trn,x_Trn,y_Trn,a_Trn,L,mu_p(I_ada(2)));
for r=1:1
[squared_error_ada,~,~,~,~,~,~,H_ada{r},~,~,~,~] = online_adaptive(A_start,x_Trn,y_Trn,a_Trn,D_trn,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_i,P,W,eta_p,eta_w,Constant);
[regret_ada_seq{r}]=regret(squared_error_ada,mu_p(I_ada(2)),se_sequence,hb);
end
plot(regret_ada_seq{r});
% regret_ada_batch(z)=mean(regret_ada_seq);
% freq_ada{z}=avg_freq(H_ada);


ax=gca;
lines = get(ax, 'Children');

% Ensure the lines are sorted in the order they were plotted
lines = flipud(lines); 

% Change the color of the lines
%set(lines(1), 'Color', 'r'); % Change the color of the first line to red
%set(lines(2), 'Color', 'b'); % Change the color of the second line to blue

% Update the legend
legend(ax, {'S-OGF','Ada-OGF'},'FontSize', 20);

% Optionally, you can also set the title and labels if not done before
xlabel(ax, 'Incoming Nodes', 'FontSize', 20);
ylabel(ax, 'Normalized cumulative regret', 'FontSize', 20);
title(ax, 'Stochastic regret for movielens', 'FontSize', 20);
grid(ax, 'on');

% Increase the thickness of the lines
set(lines, 'LineWidth', 2);

%% Online adaptive 2

% mu_p=[1e-3,1e-2];
% step_p=[1e-3,1e-2];P=10;W=10;
% rnmse_ada2=zeros(length(step_p),length(mu_p));
% eta_p=1e-4;eta_w=1e-4;
% for i=1:length(step_p)
%       for j=1:length(mu_p)
%           for r=1:1
% [~,rnmse_seq_ada2(r),~,~,~,~,~,~,~,~] = online_adaptive2(A_start,x_Trn,y_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i,P,eta_p);
%           end
%       rnmse_ada2(i,j)=mean(rnmse_seq_ada2);
%       end
%  end
%  rnmse_ada2(isnan(rnmse_ada2))=Inf;
%  [min_ada2,I_ada2] = min2d(rnmse_ada2);
%  % 
% % % Test
%  for r=1:10
% [~,~,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,y_latest_ada,~,p_bar,D] = online_adaptive2(A_start,x_Trn,y_Trn,a_Trn,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_i,P,eta_p);   
% [~,rnmse_seq_ada_test(r),~,p_f]=online_ada2_eval(A_Tst,x_Tst,y_Tst,a_Tst,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,y_latest_ada,P,eta_p,p_bar,D);
% end
% final_ada2=mean(rnmse_seq_ada_test);
% % 
% % % Regret
% [se_sequence,~,h_batch] = deterministic_batch(A_Trn,x_Trn,y_Trn,a_Trn,L,mu_p(I_ada2(2)));
% for r=1:10
% [squared_error_ada2,~,~,~,~,~,~,H_ada2{r}] = online_adaptive2(A_start,x_Trn,y_Trn,a_Trn,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_i,P,eta_p);
% regret_ada2_seq(r)=regret(squared_error_ada2,mu_p(I_ada2(2)),se_sequence,h_batch);
% end
% regret_ada2_batch=mean(regret_ada2_seq);
% freq_ada2=avg_freq(H_ada2);
% 
% 
% 
%% Online Kernel Learning
%sigma: kernel type, step: learning rate, mu: regularizer hyperparam, D: feature length
% sigma=0.1;D=(L+1)/2;step=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
% mu=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5];step=mu;
% rnmse_kernel=zeros(length(step),length(mu));
% hki = kernel_pretrain(A_Trn,x_Trn,0.5,0.1,D,L,sigma);
% for i=1:length(step)
% for j=1:length(mu)
% [squared_error_ker{i,j},rnmse_kernel(i,j),h_ker{i,j},A_latest_ker{i,j},x_latest_ker{i,j},a_latest_ker{i,j},y_latest_ker{i,j}]=online_kernel(A_Trn,x_Trn,y_Trn,a_Trn,sigma,D,step(i),mu(j),hki);
% end
% end
% rnmse_kernel(isnan(rnmse_kernel))=Inf;
% [min_ker,I_ker] = min2d(rnmse_kernel);
% 
% % Test
% [squared_error_ker_test{z},rnmse_seq_ker_test]=online_kernel_eval(A_Tst,x_Tst,y_Tst,a_Tst,sigma,D,step(I_ker(1)),mu(I_ker(2)),h_ker{I_ker(1),I_ker(2)},A_latest_ker{I_ker(1),I_ker(2)},x_latest_ker{I_ker(1),I_ker(2)},a_latest_ker{I_ker(1),I_ker(2)},y_latest_ker{I_ker(1),I_ker(2)});
% final_ker(z)=rnmse_seq_ker_test;
% median_ker(z)=median(squared_error_ker_test{z});
% % 
% % 
% %% Online multi-hop Kernel Learning
% 
% sigma=0.1;D=(L+1)/2;H=L;
% step=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
% mu=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5];
% rnmse_mhkernel=zeros(length(step),length(mu));
% hki = kernel_pretrain(A_Trn,x_Trn,0.5,0.1,D,L,sigma);
% for i=1:length(step)
% for j=1:length(mu)
% [squared_error_mhker,rnmse_mhkernel(i,j),theta_mhker{i,j},A_latest_mhker{i,j},x_latest_mhker{i,j},a_latest_mhker{i,j},y_latest_mhker{i,j}]=online_mhkernel(A_Trn,x_Trn,y_Trn,a_Trn,sigma,D,step(i),mu(j),hki,H);
% end
% end
% rnmse_mhkernel(isnan(rnmse_mhkernel))=Inf;
% [min_mhker,I_mhker] = min2d(rnmse_mhkernel);
% % 
% % %Test
% [squared_error_kermh_test{z},rnmse_seq_mhker_test]=online_mhkernel_eval(A_Tst,x_Tst,y_Tst,a_Tst,sigma,D,H,step(I_mhker(1)),mu(I_mhker(2)),theta_mhker{I_mhker(1),I_mhker(2)},A_latest_mhker{I_mhker(1),I_mhker(2)},x_latest_mhker{I_mhker(1),I_mhker(2)},a_latest_mhker{I_mhker(1),I_mhker(2)},y_latest_mhker{I_mhker(1),I_mhker(2)});
% final_mhker(z)=rnmse_seq_mhker_test;
% median_mh(z)=median(squared_error_kermh_test{z});
% % 
% 
% %% Inductive transfer
% gamma=[1e-3,1e-2,1e-1,1,10];
% for i=1:length(gamma)
% rnmse_pre_seq=zeros(10,1);
% for r=1:1%becuase of the randomness in generative filter function
% h_i= pretrained_filter(A_start,U_e,C,L,gamma(i));
% [squared_error,rnmse_pre_seq(r)] = online_pretrain(A_Trn,x_Trn,y_Trn,a_Trn,L,h_i);
% end
% rnmse_pre(i)=mean(rnmse_pre_seq);
% end
% rnmse_pre(isnan(rnmse_pre))=Inf;
% [min_pre,I_pre] = min(rnmse_pre);
% %Test
% 
% h=pretrained_filter(A_start,U_e,C,L,gamma(I_pre));
% [squared_error_pre_test{z},rnmse_seq_pre_test(z)]=online_pretrain(A_Tst,x_Tst,y_Tst,a_Tst,L,h);
% median_pre(z)=median(squared_error_pre_test{r});
% final_pre(z)=mean(rnmse_seq_pre_test);


end



