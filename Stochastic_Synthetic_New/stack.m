function A_x = stack(A,x,L,type)
%% Inputs
% A,x: Adjacency matrix and graph signal
% L: filter/ stacking order
% type: type of stacking (see below)
%% Outputs
%A_X: [0,Ax,A^2x,...,A^{L-1}x] (type = 0)
%A_X: [x,Ax,A^2x,...,A^{L}x] (type = 1)
%% Iterative computation
switch type
    case 0   
        A_x=x;prev=x;
        for i=1:L
            prev=A*prev;
            A_x=[A_x,prev];
        end
        A_x=[zeros(length(x),1),A_x(:,1:L)];
    case 1
        A_x=x;prev=x;
        for i=1:L
            prev=A*prev;
            A_x=[A_x,prev];
        end
end
end