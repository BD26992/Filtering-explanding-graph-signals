function Op = clean(R,t_u,t_i)
%% Discussion
%function to clean the user item rating matrix on a threshold basis
% all users with less than t_u ratings are discarded
% all items with less than t_i users rating them are discarded
%% Inputs
% R: ratings matrix
% t_u: user threshold
% t_i: item threshold
%% Outputs
% Op: cleaned matrix
%% Users
[U,I]=size(R);Op=R;
for u=1:U
    a(u)=length(find(R(u,:)));
end
x=find(a<t_u);
%% Items
for i=1:I
    b(i)=length(find(R(:,i)));
end
y=find(b<t_i);
%% Clean
Op(x,:)=[];Op(:,y)=[];
end