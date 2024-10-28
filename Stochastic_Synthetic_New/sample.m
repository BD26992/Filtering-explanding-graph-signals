function op = sample(p,w,N,nlinks)
%% Inputs
%p,w,nlinks: probability vector, weight vector, and number of links
%% Output
%op: attachment vector
%% Generate the sampling pattern
count=0;t=[];% t contains indexes of selected nodes
while count<=nlinks % required condition 
    seed=double(rand(N,1)<p);%select nodes by bernoulli probability.
    t=[t;find(seed)];%add nodes to t
    if length(unique(t))>=nlinks %if we have more than needed, random draw nlinks from them
        t=t(randperm(nlinks));
        break;
    else
        count=length(unique(t));%else keep updating the number of nodes
    end
end
%% Generate the vector
ber=zeros(N,1);ber(unique(t))=1;
op=ber.*w;
end