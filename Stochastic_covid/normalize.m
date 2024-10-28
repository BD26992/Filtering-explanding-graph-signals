function [A]=normalize(R_e,type)
%% Description: normalization done on R_e
%% Inputs
%% Outputs
%% Code
[N,M]=size(R_e);
switch type
    case 'row'
     for i=1:N
     mx=mean(R_e(i,:));sdx=std(R_e(i,:));
     A(i,:)=(R_e(i,:)-mx*ones(1,M))/sdx;
     end
    case 'column'
     for i=1:M
     mx=mean(R_e(:,i));sdx=std(R_e(:,i));
     A(:,i)=(R_e(:,i)-mx*ones(N,1))/sdx;
     end   
end