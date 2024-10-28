function pc = pearson(a,b)
nna=find(a); 
nnb=find(b); 
i=intersect(nna,nnb);
n_c=length(i);
p=a(i);
q=b(i);
%pc=(p-m_a*ones(n_c,1))'*(q-m_b*ones(n_c,1))/(std(p)*std(q));
covmat=corrcoef([p,q]);
if size(covmat)==[1,1]
    pc=covmat;
else
pc=covmat(2,1);
end
if isnan(pc)
    pc=0;
end
end