function [trans,thrs,coefs]=pso(strfitnessfct,trainData,trainClass,transs,show,l)
[m,n]=size(trainData);
[m,c]=size(trainClass);

numd=n;
nump=4+floor(3*log(n));
max_iter=log(n+1)*150;

omega=0.71;
c1=1.49;
c2=c1;

[x,v]=initParticles(nump,numd);
p=x;

fits = getFits(strfitnessfct, x, trainData, trainClass, transs, l);
fitsp=fits;
[best_v, best_i] = min(fitsp);
g=repmat(p(best_i,:),nump,1);
fitslist=[];
for i=1:max_iter    
    v=omega*v+c1*rand(nump,numd).*(p-x)+c2*rand(nump,numd).*(g-x);
    x=x+v;
    fits = getFits(strfitnessfct, x, trainData, trainClass, transs, l);
    im_inds = find(fits < fitsp);
    p(im_inds,:)=x(im_inds,:);
    fitsp(im_inds) = fits(im_inds);
    [best_v, best_i] = min(fitsp);
    g=repmat(p(best_i,:),nump,1);
    fitg=best_v;
    fitslist=[fitslist,fitg];
    if(show==1)
        showfunc(fitslist, trainData, trainClass, g(1,:)', transs);
    end
    
end
d=trainData*g(1,:)';
[avgf,avgMarg,thrs,coefs]=optimalMargin(d,trainClass);
trans=g(1,:)';


function [x,v]=initParticles(nump,numd)
x=randn(nump,numd)*5;
v=randn(nump,numd)*5;

function fits = getFits(strfitnessfct, x, trainData, trainClass, transs, l)
fits=zeros(size(x,1),1);

for i=1:size(x,1)
    myX=x(i,:)';
    myX=myX/norm(myX);
    fits(i) = feval(strfitnessfct, myX, trainData, trainClass, transs, l); % objective function call
end

function showfunc(fits, data, classes, bestSoFar, transs)

subplot(211);plot(fits,'-o');
title(['best so far ' num2str(fits(end))]);    
y=[bestSoFar,transs];
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