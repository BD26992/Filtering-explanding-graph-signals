function D = premake_dictionary(A_start,a_T)
T=size(a_T,2);D=cell(T,2);A_ex=A_start;
D{1}=dictionary_maken(0.5*(A_start+A_start'));
for t=2-1:T
N=size(A_ex,1);
A_ex=[A_ex,zeros(N,1);a_T{t}',0];
D{t}=dictionary_maken(0.5*(A_ex+A_ex'));
end
end