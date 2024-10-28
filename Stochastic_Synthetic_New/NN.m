function Op = NN(A,N_links,type)
%% Description
% Converts a dense fully connected adj matrix into an NN with two types of links: directed and undirected 
%% Inputs
% A: Input Dense Adj matrix
% N_links: number of desired edges for each node
%% Outputs
% Op: Output sparser Adj matrix
%% Code
[N]=size(A,1);Op=zeros(N,N);
switch type
    case 'directed'
for i=1:N
    [c,b]=sort(A(i,:),'descend');
    Op(i,b(1:N_links))=c(1:N_links);
end
    case 'undirected'
   for i=1:N
    [c,b]=sort(A(i,:),'descend');
    Op(i,b(1:N_links))=c(1:N_links);
   end 
   Op=0.5*(Op+Op');
end
Mask=ones(N,N)-eye(N);
Op=Op.*Mask;

end