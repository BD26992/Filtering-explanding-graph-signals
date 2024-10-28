function regret_statistics(e1,e2,T)
op=zeros(length(T),1);
for t=1:length(T)
    op(t)=regret(e1(1:T(t)),e2(1:T(t)));
end
end