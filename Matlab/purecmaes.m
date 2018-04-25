function [trans,thrs,coefs]=purecmaes(strfitnessfct,data,classes,transs,l,show,ini)   % (mu/mu_w, lambda)-CMA-ES
% --------------------  Initialization --------------------------------  
% User defined input parameters (need to be edited)
%   strfitnessfct = 'myfit';  % name of objective/fitness function
N = size(data,2);               % number of objective variables/problem dimension
xmean = rand(N,1);    % objective variables initial point
sigma = 1;          % coordinate wise standard deviation (step size)


stopfitness = -1e10;  % stop if fitness < stopfitness (minimization)
stopeval = 1e2*N^2;   % stop after stopeval number of function evaluations

% Strategy parameter setting: Selection  
lambda = 4+floor(3*log(N));  % population size, offspring number
stopeval = min([lambda*log(N+1)*150,stopeval]);
if(~isempty(ini))
    xmean=ini;
    sigma=0.01;
end
mu = lambda/2;               % number of parents/points for recombination
weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
mu = floor(mu);        
weights = weights/sum(weights);     % normalize recombination weights array
mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

% Strategy parameter setting: Adaptation
cc = (4+mueff/N) / (N+4 + 2*mueff/N);  % time constant for cumulation for C
cs = (mueff+2) / (N+mueff+5);  % t-const for cumulation for sigma control
c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma 
                                                  % usually close to 1
% Initialize dynamic (internal) strategy parameters and constants
pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
B = eye(N,N);                       % B defines the coordinate system
D = ones(N,1);                      % diagonal D defines the scaling
C = B * diag(D.^2) * B';            % covariance matrix C
invsqrtC = B * diag(D.^-1) * B';    % C^-1/2 
eigeneval = 0;                      % track update of B and D
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of 
                                  %   ||N(0,I)|| == norm(randn(N,1)) 
% -------------------- Generation Loop --------------------------------
counteval = 0;  % the next 40 lines contain the 20 lines of interesting code 
fits=[];
bestSoFar=[];
while counteval < stopeval

  % Generate and evaluate lambda offspring
  for k=1:lambda,
      arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C) 
      arx(:,k)=arx(:,k)/norm(arx(:,k));
%       arx(:,k)=-1+(2)*(1-cos(pi*arx(:,k)))/2;
      if(~isempty(transs))
        arx(:,k)=arx(:,k)-(arx(:,k)'*(transs*transs'/(transs'*transs)))';
      end      
%       if(~isempty(transs))
%         arx(:,k)=(arx(:,k)'*transs*transs')';
%       end
      arfitness(k) = feval(strfitnessfct, arx(:,k), data, classes, transs, l); % objective function call
      counteval = counteval+1;
  end
  
  % Sort by fitness and compute weighted mean into xmean
  [arfitness, arindex] = sort(arfitness); % minimization
  if(isempty(bestSoFar))
    bestSoFar=arx(:, arindex(1));
    bestFit=arfitness(1);
  else
      if(bestFit>arfitness(1))
          bestSoFar=arx(:, arindex(1));
          bestFit=arfitness(1);
      end
  end
  xold = xmean;
  xmean = arx(:,arindex(1:mu))*weights;   % recombination, new mean value

  % Cumulation: Update evolution paths
  ps = (1-cs)*ps ... 
        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma; 
  hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4 + 2/(N+1);
  pc = (1-cc)*pc ...
        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;

  % Adapt covariance matrix C
  artmp = (1/sigma) * (arx(:,arindex(1:mu))-repmat(xold,1,mu));
  C = (1-c1-cmu) * C ...                  % regard old matrix  
       + c1 * (pc*pc' ...                 % plus rank one update
               + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
       + cmu * artmp * diag(weights) * artmp'; % plus rank mu update

  % Adapt step size sigma
  sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1)); 

  % Decomposition of C into B*diag(D.^2)*B' (diagonalization)
  if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
      eigeneval = counteval;
      C = triu(C) + triu(C,1)'; % enforce symmetry
      [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
      D = sqrt(diag(D));        % D is a vector of standard deviations now
      invsqrtC = B * diag(D.^-1) * B';
  end

  % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable 
  if arfitness(1) <= stopfitness || max(D) > 1e7 * min(D)
      break;
  end

  if(show>0)
    fits=[fits,bestFit];
    showfunc(fits, data, classes, bestSoFar, transs);
  end
end % while, end generation loop

xmin = bestSoFar; % Return best point of last iteration.
                         % Notice that xmean is expected to be even
                         % better.
[avgf,avgMarg,thrs,coefs] = feval(strfitnessfct, xmin, data, classes, transs, l); % objective function call

trans=xmin;

% ---------------------------------------------------------------  

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