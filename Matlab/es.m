function [trans,thrs,coefs,bestFit]=es(strfitnessfct,data,classes,transs,show,ini,l)  

% lambda=20;
% mu=10;
numVar=size(data,2);
lambda = 4+floor(3*log(numVar));  % population size, offspring number
mu=floor(lambda/2);

maxgen=log(numVar+1)*150;

sigma=1*ones(1,mu);
tau=(1/(sqrt(2*numVar)));
if(isempty(ini))
    indiv=2*(rand(mu,numVar)-0.5);
else
    indiv=2*(rand(mu,numVar)-0.5);
    indiv(1,:)=ini';
    sigma=.1*ones(1,mu);
end

for i=1:mu
    indiv(i,:)=indiv(i,:)/norm(indiv(i,:));
end

fitnessindi=fitness(strfitnessfct, indiv, data, classes, transs, l);
fitslist=[];
it=1;
bestSoFar=[];

for i=1:maxgen
    [v,indx]=min(fitnessindi);
    
    if(isempty(bestSoFar))
        bestSoFar=indiv(indx,:);
        bestFit=v;
    else
        if(bestFit>v)
            bestSoFar=indiv(indx,:);
            bestFit=v;
        end
    end
    
    recsigma=mean(sigma);
    recindi=mean(indiv,1);

    sigmaoff=recsigma * (exp(tau*randn(lambda,1)));
    offs = repmat(sigmaoff,1,numVar).*randn(lambda,numVar) + repmat(recindi,lambda,1);
    
    for j=1:lambda
        offs(j,:)=offs(j,:)/norm(offs(j,:));
    end
    fitnessoff=fitness(strfitnessfct, offs, data, classes, transs, l);  
    [~,order]=sort(fitnessoff);
    indiv=offs(order(1:mu),:);
    sigma=sigmaoff(order(1:mu));
    fitnessindi=fitnessoff(order(1:mu));   
    
    it=it+1;
    
    if(show>0)
        fitslist=[fitslist,bestFit];
        showfunc(fitslist, data, classes, bestSoFar, transs);
    end
end
xmin = bestSoFar'; % Return best point of last iteration.
                         % Notice that xmean is expected to be even
                         % better.
[avgf,avgMarg,thrs,coefs] = feval(strfitnessfct, xmin, data, classes, transs, l); % objective function call

trans=xmin;
end

function fits=fitness(strfitnessfct, indivs, data, classes, transs, l)
    for i=1:size(indivs,1)
        fits(i)=feval(strfitnessfct, indivs(i,:)', data, classes, transs, l); % objective function call
    end
end

function showfunc(fits, data, classes, bestSoFar, transs)

subplot(211);plot(fits,'-o');
title(['best so far ' num2str(fits(end))]);    
y=[bestSoFar,transs]';
td=data*y;
subplot(212);
if(size(y,2)==1)
    for i=1:size(classes,2)
        histogram(td(classes(:,i)==1),20);
        hold on;
    end    
    hold off;
end
if(size(y,2)==2)
    gscatter(td(:,1),td(:,2),vec2ind(classes')',[],[],10);
end
if(size(y,2)==3)
    scatter3(td(:,1),td(:,2),td(:,3),[],vec2ind(classes')');
end

drawnow;
end