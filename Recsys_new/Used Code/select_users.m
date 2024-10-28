function [A,B]=select_users(R,selection,N_start)
%% Description
% Function that selects some users and splits the user-item matrix
%% Inputs
% R: User item matrix
% selection: selection type
% N_start: Number of starting users to be selected
%% Outputs
% A: N_sel times I matrix of ratings of selected users
% A: N-N_sel times I matrix of ratings of selected users
%% Code
[U,~]=size(R);
switch selection
    case 'random'
        t=randperm(U);
        a=t(1:N_start);b=t(N_start+1:end);
        A=R(a,:);B=R(b,:);
    case 'max ratings'
        for i=1:U
            a(i)=length(find(R(i,:)));
        end
        [~,b]=sort(a,'descend');
        selected_users=b(1:N_start);A=R(selected_users,:);
        online_users=b(N_start+1:end);
        t=randperm(length(online_users));
        online_users_shuffled=online_users(t);
        B=R(online_users_shuffled,:);
end
end