function Op=similarity(X,Y,type)
%% Description
% Outputs similarity matrix following similarity type as provided as input
% between elements of two matrices
%% Inputs
% X: matrix 1 (observations row-wise)
% Y: matrix 2 (observations row-wise)
%% Outputs
% Op: similarity matrix
%% Code
[M,~]=size(X);[N,~]=size(Y);Op=zeros(M,N);
for i=1:M
for j=1:N
        switch type
            case 'cosine'
        Op(i,j)=X(i,:)*Y(j,:)'/(norm(X(i,:))*norm(Y(j,:)));
            case 'pearson'
        Op(i,j)=pearson(X(i,:),Y(j,:));
            case 'gaussian'
        Op(i,j)=exp(-(norm(X(i,:)-Y(i,:))^2/10));
         %%write expression here for pearson
        end
end
end
end