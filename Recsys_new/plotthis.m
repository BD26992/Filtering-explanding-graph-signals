DOGF=unpack_cell(squared_error_det_test,4);
SOGF=unpack_cell(squared_error_stop_test,4);
ADA=unpack_cell(squared_error_ada_test,5);
OKL=unpack_cell(squared_error_ker_test,4);
OMHKL=unpack_cell(squared_error_kermh_test,4);
XXXX=[DOGF,SOGF,ADA(1:2533),OKL,OMHKL];
boxplot(XXXX, 'Labels', {'D-OGF', 'S-OGF', 'Ada-OGF','OKL','OMHKL'});
grid on
set(gca,'FontSize',16)