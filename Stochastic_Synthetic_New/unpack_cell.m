function y = unpack_cell(x,N)
y=[];
for n=1:N
    y=[y;x{n}];
end
end