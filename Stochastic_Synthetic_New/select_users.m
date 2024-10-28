function [A,B,a,b]=select_users(R,selection,N_start)
%% Description
% Function that selects some nodes and splits the node-time matrix
%% Inputs
% R: node item matrix
% selection: selection type
% N_start: Number of starting nodes to be selected
%% Outputs
% A: N_sel times I matrix of ratings of selected nodes
% B: N-N_sel times I matrix of ratings of selected nodes
% a: identities of selected nodes
% b: identities of incoming nodes
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
        selected_users=b(1:N_start);
        online_users=b(N_start+1:end);
        A=R(selected_users,:);
        B=R(online_users,:);
end
end