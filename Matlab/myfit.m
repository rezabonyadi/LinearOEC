function [f,marg,thr,coef]=myfit(x,data,classes,transs, l)
    d=data*x;
    [f,marg,thr,coef]=optimalMargin(d,classes);
    f=f+l(1)*sum(abs(x));
    if(~isempty(transs))
        a=1;% Lagrangian
        f=f+a*sum(abs(transs'*x));
    end
    if(f==0)
        f=f-marg;
    end
end
function [thr,coef,perf,marg]=optimalDisc1D(d,c)
% d=gpuArray(d);
    [ds,di]=sort(d);
    cs=c(di,1);
    t1=sum(cs);
    t0=length(cs)-t1;

    thr=0;
    coef=0;marg=0;
    n0l=0;
    n1l=0;
    perf=0;

    for i=1:length(ds)-1
        if(cs(i)==0)
            n0l=n0l+1;
        else
            n1l=n1l+1;
        end
    %     acc1=(n0l/t0)*(1-n1l/t1); % 0s to the lesft, 1s to the right
    %     acc2=(n1l/t1)*(1-n0l/t0); % 1s to the left, 0s to the right
        acc1=(n0l/t0)+(1-n1l/t1); % 0s to the lesft, 1s to the right
        acc2=(n1l/t1)+(1-n0l/t0); % 1s to the left, 0s to the right
        if(acc1>perf)
            thr=-((ds(i)+ds(i+1))/2);
            coef=-1;
            perf=acc1;
            marg=abs(ds(i)-ds(i+1));
        end
        if(acc2>perf)
            thr=(ds(i)+ds(i+1))/2;
            coef=1;
            perf=acc2;
            marg=abs(ds(i)-ds(i+1));
        end
    end
end
function [avgf,avgMarg,thr,coef]=optimalMargin(d,classes)
    avgf=0;avgMarg=0;    
    [thr,coef,f,marg]=optimalDisc1D(d,classes);
    f=f/2;
    avgf=avgf+(1-f);avgMarg=avgMarg+marg;
    % avgf=avgf/((numC*(numC-1))/2);
    % avgMarg=avgMarg/((numC*(numC-1))/2);
end