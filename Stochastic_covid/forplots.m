
X_det=[];X_sto=[];X_ada=[];X_ker=[];X_kermh=[];
for
er=[X_ker;squared_error_ker_test{r}'];
X_kermh=[X_kermh;squared_error_kermh_test{r}'];
end

figure
violinplot([X_det, X_sto, X_ada,X_ker,X_kermh],{'Det','Sto','Ada','Ker','Mhker'})


for r=1:10
figure
violinplot([squared_error_det_test{r}', squared_error_psto_test{r}', error_seq_ada_test{r}',squared_error_ker_test{r}',squared_error_kermh_test{r}'],{'Det','Sto','Ada','Ker','Mhker'})
end