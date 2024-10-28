function plot_freq_response(plottype,freq_batch,H_det,freq_sto,H_psto,freq_ada,freq_ada2)
switch plottype
    case 'multiple'
        %% Det plot
        figure
        plot(freq_batch,'linewidth',2,'r');
        hold on;
        plot(H_det);
        xlabel('Frequency');ylabel('Response');title('Deterministic Freq. Response');
        grid on;
        
        %% Sto plot
        figure
        plot(freq_batch,'linewidth',2,'r');
        hold on;
        plot(freq_sto);
        xlabel('Frequency');ylabel('Response');title('Approx Stochastic Freq. Response');
        grid on;
        
        %% Pure Sto plot
        figure
        plot(freq_batch,'linewidth',2,'r');
        hold on;
        plot(H_psto);
        xlabel('Frequency');ylabel('Response');title('Pure Stochastic Freq. Response');
        grid on;
        
        %% Ada plot
        figure
        plot(freq_batch,'linewidth',2,'r');
        hold on;
        plot(freq_ada);
        xlabel('Frequency');ylabel('Response');title('Adaptive Freq. Response');
        grid on;
        
        
        %% Ada2 plot
        figure
        plot(freq_batch,'linewidth',2,'r');
        hold on;
        plot(freq_ada);
        xlabel('Frequency');ylabel('Response');title('Adaptive Freq. Response');
        grid on;
        
       
    case 'single'
        
        figure
        gridplot=[-1:0.01:1];hold on;
        plot(gridplot,H_det(:,end),'linewidth',1);
        plot(gridplot,H_psto(:,end),'linewidth',1);
        plot(gridplot,freq_sto(:,end)','linewidth',1);
        plot(gridplot,freq_ada(:,end)','linewidth',1);
        plot(gridplot,freq_batch','linewidth',2);
        %plot(gridplot,freq_ada2(:,end)');
        xlabel('Frequency');ylabel('Response');title('Day 265');
        legend('Det.','Sto','ApSto','AdaSto','Batch');
        grid on;
        ax=gca;
        ax.FontSize=18;
end
end