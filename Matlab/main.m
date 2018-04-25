clear all;
close all;

% Breast Cancer dataset: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
% data=readtable('../Data/BC.csv'); 

% HillVally: http://archive.ics.uci.edu/ml/datasets/Hill-Valley
data=readtable('../Data/HV.csv');

% ATTENTION: In both datasets, the null space has been removed and the data 
% has been normalized to 0 mean and 1 variance. This makes the HV problem 
% harder to solve!

x=table2array(data);
y=x(:,end);
x=x(:,1:end-1);


% With SVM initialization
% svm = fitcsvm(x,y);
% model=fitoec(size(x,2),'optimizer','cmaes','show',0,'regul',0,'ini',svm.Beta);

% No initialization
model=fitoec(size(x,2),'optimizer','cmaes','show',0,'regul',0,...
    'ini',[]);

% Optimize the model (fit)
tic;
model = model.optimise(x,y);
time = toc;

% Evaluate the model on some data, x
y_hat=predict(model,x);

% Display the loss values and visualize the discriminator
loss = sum(abs(y-y_hat));
disp(['Total loss OEC: ' num2str(loss) ', time: ' num2str(time*1000) ' (ms)']);
model.visualise(x,y);
