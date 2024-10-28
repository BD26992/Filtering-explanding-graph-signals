
clear all
%script for running simulations on synthetic data
%% How is the regret obtained?
% 1. It is only possible for online, stochastic, stochastic pure, ada, and ada2
% 2. We evaluate all regret over the training set only.
% 2. Not for kernels because they have a different batch solution
% altogether
% 3. Not for inductive transfer as it is not an online algorithm
% To calculate the regret w.r.t to the optimal batch solution, we first
% 1. Obtain the fair batch solution, corresponding to the same mu as the
% online method being compared
% 2. Obtain the regret between them. at each time, the cost is
% l(t)=fitting(t)+mu*||h_{t-1}||^2. We evaluate this for both methods,
% subtract and normalize them over the sequence length.


%% Note that RNMSE and the Regret are different.
% RNMSE is involved in the regret calculation as it is involved with a part of the cost,
% i.e., the fitting term. But it has significance in itself, so we also
% obtain that

%% For all stochastic methods
% 1. During the test evaluation, we train again and continue the latest
%    filter into the test set with the selected hyperparameters.
% 2. During regret calculation, we train again and evaluate the regret
%    multiple times, due to the stochastic nature. we aggregate the mean of
%    all regret statistics in the end.

%% How to select hyperparameters
% They are selected as follows:
% 1. For each hyperparameter perform online learning over the training set
% 2. Retain one with min rnmse (or one with the rnmse equivalent of total loss?)
% 3. Perform test using that hyperparameter.

%% Frequency Plots
% For each online method, they are plotted as follows:
% There are two options for plots
% In the first type, one plot for each method
% In the second, one plot for all

%% Generate data

load mean_data.mat
% epsilon=1;
% N=100;p=(log(N)*(1+epsilon))/N;
% A=gsp_erdos_renyi(N,0.2);
% weight_mask=rand(N,N);
% weight_mask=0.5*(weight_mask+weight_mask');
% A_start=full(double(A.W)).*weight_mask;
% A_start=A_start/abs(max(eig(A_start)));
for g=1:5
% gen='kernel';%2 options: 'mean' and 'filter'
% rule='uniform';N_links=5;T=200;lim=10000;gamma=1;frac=0.8;L=5;ptfrac=0.8;
% switch gen
%     case 'filter'
%         bl=3;[U,V]=eig(diag(sum(A_start,2))-A_start);
%         x_start=10*U(:,1:bl)*rand(bl,1);
%         x_start=(x_start-mean(x_start)*ones(N,1))/std(x_start);
%         h = generative_filter(A_start,x_start,L,gamma,ptfrac);
%         [A_Trn,x_Trn,a_Trn,D_Trn,A_Tst,x_Tst,a_Tst,D_Tst] =  build_synthetic_composite(N_links,A_start,x_start,T,h,L,gen,frac);
%     case 'mean'
%         bl=10;[U,V]=eig(full(A.L));
%         x_start=10*U(:,1:bl)*rand(bl,1);
%         h = generative_filter(A_start,x_start,L,gamma,ptfrac);
%         [A_Trn,x_Trn,a_Trn,D_Trn,A_Tst,x_Tst,a_Tst,D_Tst] = build_synthetic_composite(N_links,A_start,x_start,T,h,L,gen,frac);
% 
%     case 'kernel'
%         variance=1;
%         [A_Trn,x_Trn,a_Trn,D_Trn,A_Tst,x_Tst,a_Tst,D_Tst] = build_synthetic_kernel(rule,N_links,A_start,T,variance,frac);
%         x_start=x_Trn{1};
% end


%Z=[5];
%% Batch filter
%for z=1:length(Z)
L=5;
% gamma_batch=[1e-3,1e-2,1e-1,1,10];
% rnmse_batch=zeros(length(gamma_batch),1);
% for i=1:length(gamma_batch)
% [~,rnmse_batch(i),h_batch{i}] = batch(A_Trn,x_Trn,a_Trn,L,gamma_batch(i));
% end
% rnmse_batch(isnan(rnmse_batch))=Inf;
% [min_batch,I_batch] = min(rnmse_batch);
% final_batch(g)=batch_eval(A_Tst,x_Tst,a_Tst,L,h_batch{I_batch});
% % batch frequency response
% grid=[-1:0.01:1]';
% vandermonde=fliplr(vander(grid));
% freq_batch(:,z)=abs(vandermonde(:,1:L+1)*h_batch{I_batch});
% 
%    
% 
% 
% 
%% Online Learning
%
%scale=1;
% %mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5];step_p=mu_p;
%mu_p=1e-3;step_p=[1e-5,1e-4,1e-3,1e-1,1,5,6];
h_i = generative_filter(A_start,x_start,L,10,ptfrac);
%  squared_error_det=cell(length(step_p),length(mu_p));
%  rnmse_det=zeros(length(step_p),length(mu_p));
% for i=1:length(step_p)
%        for j=1:length(mu_p)
%             [squared_error_det{i,j},rnmse_det(i,j),h_det{i,j},A_latest_det{i,j},x_latest_det{i,j},a_latest_det{i,j},H_det{i,j}]=online_proposed(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),h_i,scale);
%        [se_sequence{i},~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(j));
% [~,regret_det_batch(i,:)]=regret(squared_error_det{i,j},mu_p(j),se_sequence{i},hb);
%        end
% end
% figure
% plot(regret_det_batch');
% ax=gca;
% lines = get(ax, 'Children');
% 
% % Ensure the lines are sorted in the order they were plotted
% lines = flipud(lines); 
% 
% % Change the color of the lines
% %set(lines(1), 'Color', 'r'); % Change the color of the first line to red
% %set(lines(2), 'Color', 'b'); % Change the color of the second line to blue
% 
% % Update the legend
% legend(ax, {'$\eta=10^{-5}$', '$\eta=10^{-4}$','$\eta=10^{-3}$','$\eta=10^{-1}$','$\eta=1$','$\eta=5$'}, 'Interpreter','latex','FontSize', 20);
% 
% % Optionally, you can also set the title and labels if not done before
% xlabel(ax, 'Incoming Nodes', 'FontSize', 20);
% ylabel(ax, 'Normalized cumulative regret', 'FontSize', 20);
% title(ax, 'Effect of learning rate on mean data', 'FontSize', 20);
% grid(ax, 'on');
% 
% % Increase the thickness of the lines
% set(lines, 'LineWidth', 2);
% xlim([1,158])
% rnmse_det(isnan(rnmse_det))=Inf;
% [min_det,I_det] = min2d(rnmse_det);
% % 
% % 
% % % Test set evaluation
% [squared_error_det_test,rnmse_seq_det_test,h_det_test]=online_proposed_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_det(1)),mu_p(I_det(2)),h_det{I_det(1),I_det(2)},A_latest_det{I_det(1),I_det(2)},x_latest_det{I_det(1),I_det(2)},a_latest_det{I_det(1),I_det(2)});
% final_det(g)=rnmse_seq_det_test;
% std_det(g)=std(rnmse_seq_det_test);
% % Regret
%[se_sequence,~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_det(2)));
%[~,regret_det_batch]=regret(squared_error_det{I_det(1),I_det(2)},mu_p(I_det(2)),se_sequence,hb);
% % % freq_det{z}=H_det{I_det(1),I_det(2)};
% % % % 
% plot(regret_det_batch)

%% Online Stochastic Learning

% mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5];
% step_p=mu_p;rnmse_sto=zeros(length(step_p),length(mu_p));
% for i=1:length(step_p)
%      for j=1:length(mu_p)
%          rnmse_seq_sto=zeros(10,1);
%          for r=1:10
%              [~,rnmse_seq_sto(r),~,~,~,~,~]=online_stochastic(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i);
%          end
%      rnmse_sto(i,j)=mean(rnmse_seq_sto);
%      end
%  end
%  rnmse_sto(isnan(rnmse_sto))=Inf;
%  [min_sto,I_sto] = min2d(rnmse_sto);
% 
% % Test set performance
% for r=1:10
% [~,~,h_sto,A_latest_sto,x_latest_sto,a_latest_sto,~]=online_stochastic(A_Trn,x_Trn,a_Trn,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_i);
% [~,rnmse_seq_sto_test(r),~]=online_stochastic_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_sto,A_latest_sto,x_latest_sto,a_latest_sto);
% end
% final_sto(z)=mean(rnmse_seq_sto_test);
% 
% % Regret calculation
%  [se_sequence,~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_sto(2)));
%  for r=1:10
%  [squared_error_sto,~,~,~,~,~,H_sto{r}]=online_stochastic(A_Trn,x_Trn,a_Trn,L,step_p(I_sto(1)),mu_p(I_sto(2)),N_links,h_i);
%  [~,regret_sto_seq{r}]=regret(squared_error_sto,mu_p(I_sto(2)),se_sequence,hb);
%  figure;
%  plot(regret_sto_seq{r}); 
% end
% regret_sto_batch(z)=mean(regret_sto_seq);
% freq_sto{z}=avg_freq(H_sto);

% 
% 
%% Online pure stochastic Learning
% 
% mu_p=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10];scale=1;
% step_p=mu_p;
% for i=1:length(step_p)
%       for j=1:length(mu_p)
%          [squared_error_psto{i,j},rnmse_psto(i,j),h_psto{i,j},A_latest_psto{i,j},x_latest_psto{i,j},a_latest_psto{i,j},H_psto{i,j}]=online_stochastic_pure(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i,scale);
%       end
% 
% end
% rnmse_psto(isnan(rnmse_psto))=Inf;
% [min_psto,I_psto] = min2d(rnmse_psto);
% % 
% % % Test set
%  [squared_error_psto_test,rnmse_seq_psto_test,h_psto_test]=online_stochastic_pure_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_psto(1)),mu_p(I_psto(2)),N_links,h_psto{I_psto(1),I_psto(2)},A_latest_psto{I_psto(1),I_psto(2)},x_latest_psto{I_psto(1),I_psto(2)},a_latest_psto{I_psto(1),I_psto(2)},scale);
%  final_psto(g)=mean(rnmse_seq_psto_test);
%  %std_psto(z)=std(rnmse_seq_psto_test);
% 
% % Regret
%  [se_sequence,~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_psto(2)));
%  [~,regret_psto_batch]=regret(squared_error_psto{I_psto(1),I_psto(2)},mu_p(I_psto(2)),se_sequence,hb);
 % plot(regret_psto_batch)
 %freq_psto{z}=H_psto{I_psto(1),I_psto(2)};
% 

%% Online adaptive
% scale=1;
% mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
% step_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];P=5;W=1;
% rnmse_ada=zeros(length(step_p),length(mu_p));
% eta_p=10;eta_w=10;
% for i=1:length(step_p)
% for j=1:length(mu_p)
% for r=1:10
% [~,rnmse_seq_ada(r),~,~,~,~,~,p_t{i,j},w_t{i,j},~,~] = online_adaptive(A_Trn,x_Trn,a_Trn,D_Trn,L,step_p(i),mu_p(j),N_links,h_i,P,W,eta_p,eta_w,scale);
% end
% rnmse_ada(i,j)=mean(rnmse_seq_ada);
% end
% end
% rnmse_ada(isnan(rnmse_ada))=Inf;
% [min_ada,I_ada] = min2d(rnmse_ada);
% 
% % Test set
% for r=1:10
% [~,~,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,~,p_bar,w_bar,D,E] = online_adaptive(A_Trn,x_Trn,a_Trn,D_Trn,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_i,P,W,eta_p,eta_w,scale);
% [check{r},rnmse_seq_ada_test(r),~]=online_ada_eval(A_Tst,x_Tst,a_Tst,D_Tst,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_ada,A_latest_ada,x_latest_ada,a_latest_ada,P,W,eta_p,eta_w,p_bar,w_bar,D_Trn{end},E,scale);
% end
% final_ada(g)=mean(rnmse_seq_ada_test);
% %std_ada(z)=std(rnmse_seq_ada_test);
% 
% 
% % Regret
% [se_sequence,~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_ada(2)));
% for r=1:1
% [squared_error_ada,~,~,~,~,~,H_ada{r},~,~,~,~]=online_adaptive(A_Trn,x_Trn,a_Trn,D_Trn,L,step_p(I_ada(1)),mu_p(I_ada(2)),N_links,h_i,P,W,eta_p,eta_w,scale);
% [~,regret_ada_seq{r}]=regret(squared_error_ada,mu_p(I_ada(2)),se_sequence,hb);
% % figure;
% % plot(regret_ada_seq{r});
% end
% regret_ada_batch(z)=mean(regret_ada_seq);
% freq_ada{z}=avg_freq(H_ada);

%% Online Adaptive 2 (update only the probability of attachment)
% scale=1;
% mu_p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
% step_p=mu_p;P=5;W=1;
% rnmse_ada2=zeros(length(step_p),length(mu_p));
% eta_p=10;eta_w=10;
% for i=1:length(step_p)
%     for j=1:length(mu_p)
%         for r=1:10
% [~,rnmse_seq_ada2(r),~,~,~,~,~,p_bar2{i,j},~] = online_adaptive2(A_Trn,x_Trn,a_Trn,D_Trn,L,step_p(i),mu_p(j),N_links,h_i,P,eta_p,scale);
%         end
%     rnmse_ada2(i,j)=mean(rnmse_seq_ada2);
%     end
% end
% rnmse_ada2(isnan(rnmse_ada2))=Inf;
% [min_ada2,I_ada2] = min2d(rnmse_ada2);
% 
% % Test set
% for r=1:10
% [~,~,h_ada2,A_latest_ada2,x_latest_ada2,a_latest_ada2,~,p_bar,D] = online_adaptive2(A_Trn,x_Trn,a_Trn,D_Trn,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_i,P,eta_p,scale);
% [~,rnmse_seq_ada_test2(r),~]=online_ada2_eval(A_Tst,x_Tst,a_Tst,D_Tst,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_ada2,A_latest_ada2,x_latest_ada2,a_latest_ada2,P,eta_p,p_bar,D_Trn{end},scale);
% end
% final_ada2(g)=mean(rnmse_seq_ada_test2);
% %std_ada2(z)=std(rnmse_seq_ada_test2);
% % % 
% % % 
% % % % Regret
% [se_sequence,~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_ada2(2)));
% for r=1:1
% [squared_error_ada2,~,~,~,~,~,H_ada2{r}]=online_adaptive2(A_Trn,x_Trn,a_Trn,D_Trn,L,step_p(I_ada2(1)),mu_p(I_ada2(2)),N_links,h_i,P,eta_p,scale);
% [~,regret_ada_seq2{r}]=regret(squared_error_ada2,mu_p(I_ada2(2)),se_sequence,hb);
% end
% % regret_ada2_batch(z)=mean(regret_ada_seq2);
% % freq_ada2{z}=avg_freq(H_ada2);

%% PC-OGF
% 
mu_p=[1e-5,1e-4,1e-3];scale=1;
step_p=mu_p;
for i=1:length(step_p)
      for j=1:length(mu_p)
         [squared_error_pc{i,j},rnmse_pc(i,j),h_pc{i,j},A_latest_pc{i,j},x_latest_pc{i,j},a_latest_pc{i,j},H_pc{i,j}]=onlinepc(A_Trn,x_Trn,a_Trn,L,step_p(i),mu_p(j),N_links,h_i,scale);
      end

end
rnmse_pc(isnan(rnmse_pc))=Inf;
[min_pc,I_pc] = min2d(rnmse_pc);
% 
% % Test set
 % [squared_error_pc_test,rnmse_seq_pc_test,h_pc_test]=online_pc_eval(A_Tst,x_Tst,a_Tst,L,step_p(I_pc(1)),mu_p(I_pc(2)),N_links,h_pc{I_pc(1),I_pc(2)},A_latest_pc{I_pc(1),I_pc(2)},x_latest_pc{I_pc(1),I_pc(2)},a_latest_pc{I_pc(1),I_pc(2)},scale);
 % final_pc(g)=mean(rnmse_seq_pc_test);
 %std_pc(z)=std(rnmse_seq_pc_test);

% Regret
 [se_sequence,~,hb] = batch(A_Trn,x_Trn,a_Trn,L,mu_p(I_pc(2)));
 [~,regret_pc_batch]=regret(squared_error_pc{I_pc(1),I_pc(2)},mu_p(I_pc(2)),se_sequence,hb);
 % plot(regret_psto_batch)
 %freq_psto{z}=H_psto{I_psto(1),I_psto(2)};
% 


%% Online Kernel Learning

%sigma: kernel type, step: learning rate, mu: regularizer hyperparam, D: feature length
%  variance=1;
%  sigma=variance;D=(L+1)/2;step=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
%  mu=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5];step=mu;
%  rnmse_kernel=zeros(length(step),length(mu));
%  hki = kernel_pretrain(A_Trn,x_Trn,0.5,0.1,D,L,sigma);
%  for i=1:length(step)
%  for j=1:length(mu)
%  [squared_error_ker,rnmse_seq_ker,h_ker{i,j},A_latest_ker{i,j},x_latest_ker{i,j},a_latest_ker{i,j}]=online_kernel(A_Trn,x_Trn,a_Trn,sigma,D,step(i),mu(j),hki);
%  rnmse_kernel(i,j)=mean(rnmse_seq_ker);
%  end
%  end
%  rnmse_kernel(isnan(rnmse_kernel))=Inf;
%  [min_ker,I_ker] = min2d(rnmse_kernel);
% % 
% % % Test
%  [~,rnmse_seq_ker_test,h_ker_test]=online_kernel_eval(A_Tst,x_Tst,a_Tst,sigma,D,step(I_ker(1)),mu(I_ker(2)),h_ker{I_ker(1),I_ker(2)},A_latest_ker{I_ker(1),I_ker(2)},x_latest_ker{I_ker(1),I_ker(2)},a_latest_ker{I_ker(1),I_ker(2)});
%  final_ker(g)=rnmse_seq_ker_test;


%% Online multi-hop Kernel Learning

%  sigma=1;D=(L+1)/2;H=L;
%  step=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1];
%  mu=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5];
%  rnmse_mhkernel=zeros(length(step),length(mu));
%  hki = kernel_pretrain(A_Trn,x_Trn,0.5,0.1,D,L,sigma);
%  for i=1:length(step)
%  for j=1:length(mu)
%  [squared_error_mhker{i,j},rnmse_mhkernel(i,j),h_mhker{i,j},A_latest_mhker{i,j},x_latest_mhker{i,j},a_latest_mhker{i,j}]=online_mhkernel(A_Trn,x_Trn,a_Trn,sigma,D,step(i),mu(j),hki,H);
%  end
%  end
%   rnmse_mhkernel(isnan(rnmse_mhkernel))=Inf;
%  [min_mhker,I_mhker] = min2d(rnmse_mhkernel);
% % 
% % % Test
%  [~,rnmse_seq_mhker_test,h_mhker_test]=online_mhkernel_eval(A_Tst,x_Tst,a_Tst,sigma,D,H,step(I_mhker(1)),mu(I_mhker(2)),h_mhker{I_mhker(1),I_mhker(2)},A_latest_mhker{I_mhker(1),I_mhker(2)},x_latest_mhker{I_mhker(1),I_mhker(2)},a_latest_mhker{I_mhker(1),I_mhker(2)});
%  final_mhker(g)=rnmse_seq_mhker_test;


%% Inductive transfer

%  gamma=[1e-3,1e-2,1e-1,1,10];
%  for i=1:length(gamma)
%  rnmse_pre_seq=zeros(10,1);
%  for r=1:10
%  h_i=generative_filter(A_start,x_start,L,gamma(i),ptfrac);
%  [squared_error,rnmse_pre_seq(r)] = online_pretrain(A_Trn,x_Trn,a_Trn,L,h_i);
%  end
%  rnmse_pre(i)=mean(rnmse_pre_seq);
%  end
%  rnmse_pre(isnan(rnmse_pre))=Inf;
%  [min_pre,I_pre] = min(rnmse_pre);
% % 
% % %Test
%  for r=1:10
%  h=generative_filter(A_start,x_start,L,gamma(I_pre),ptfrac);    
%  [~,rnmse_seq_pre_test(r)]=online_pretrain(A_Tst,x_Tst,a_Tst,L,h);
%  end
%  final_pre(g)=mean(rnmse_seq_pre_test);



%%

% final_mean(g) = online_mean(A_Tst,x_Tst,x_Trn{end});
%end
% G_batch{g}=final_batch;
% G_det{g}=final_det;
%G_psto{g}=final_psto;
% G_ada{g}=final_ada;
% G_ada2{g}=final_ada2;
% G_ker{g}=final_ker;
% G_mhker{g}=final_mhker;
% G_pre{g}=final_pre;
%G_mean{g}=final_mean;
end

%plot_freq_response('single',freq_batch(:,end),H_det{I_det(1),I_det(2)},freq_sto{end},H_psto{I_psto(1),I_psto(2)},freq_ada{end},freq_ada2{end})
% figure;hold on;
% plot([1,3,5,7,9],mean(unpack_cell(G_det,10)),'linewidth',2);
% plot([1,3,5,7,9],mean(unpack_cell(G_psto,10)),'linewidth',2);
% plot([1,3,5,7,9],mean(unpack_cell(G_ada,10)),'linewidth',2);
% plot([1,3,5,7,9],mean(unpack_cell(G_ada2,10)),'linewidth',2);
% plot([1,3,5,7,9],mean(unpack_cell(G_batch,10)),'linewidth',2);
% plot([1,3,5,7,9],mean(unpack_cell(G_pre,10)),'linewidth',2);grid on;
% legend('Det','Sto','Ada','Ada2','Batch','Pre');set(gca, 'FontSize', 20)
% ylabel('NRMSE');xlabel('Filter order');title('Filter Data');xlim([1,9])
% xticks([1,3,5,7,9]);

% figure;
% plot(regret_psto_batch,'linewidth',2);hold on;
% plot(regret_ada_seq{1},'linewidth',2);
% xlabel('Incoming Nodes','Fontsize',20);
% ylabel('Normalized cumulative regret','Fontsize',20);
% title('Stochastic regret for kernel data','Fontsize',20);
% legend('S-OGF','Ada-OGF','Fontsize',20);
% grid on;
% xlim([1 398])
