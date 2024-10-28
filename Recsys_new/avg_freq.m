function op = avg_freq(H)
R=size(H,1);
op=zeros(size(H{1}));
for r=1:R
    op=op+H{r};
end
op=op/R;
end