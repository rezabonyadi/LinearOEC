% This is a python implementation of the linear Optimal-margin Evolutionary 
% Classifier (OEC). OEC only supports linear kernels and uses derivative-free 
% optimization to optimize the 0-1 loss. Hence, it is the most rebust 
% classifier to outlayers and provides closest possible solutions to the 
% optimal 0-1 loss solution within a practical timeframe. It is up to 20 
% times slower than state-of-the-art methods (SVM, LDA, etc.), but provides 
% significantly better solutions with much better generalization ability. 
% Current implementation only supports CMAES and ES for optimization purposes.
% 
% 
% Written by M.R.Bonyadi (rezabny@gmail.com)

classdef fitoec
    
    properties (Access = public)
      optimisationSettings;
      optimisationOutcome;
      showProgress;
    end
    methods (Access = public)
        function obj=fitoec(numDimensions,varargin)
            obj.optimisationOutcome.weights=2*rand(numDimensions,1)-1;
            if(~isempty(varargin))
                allArgs=reshape(varargin,2,length(varargin)/2);
                vars=allArgs(1,:);
                vals=allArgs(2,:);
            else
                vars=[];
            end
            
            k=strmatch('optimizer',vars);
            if(~isempty(k))
                obj.optimisationSettings.optimizer=vals{k};
                if(isempty(obj.optimisationSettings.optimizer))
                    obj.optimisationSettings.optimizer='cmaes';
                end
            else
                obj.optimisationSettings.optimizer='cmaes';
            end      
            k=strmatch('show',vars);
            if(~isempty(k))
                obj.optimisationSettings.show=vals{k};
            else
                obj.optimisationSettings.show=0;
            end      
            k=strmatch('regul',vars);
            if(~isempty(k))
                obj.optimisationSettings.l=vals{k};
            else
                obj.optimisationSettings.l=0.0;
            end      
            k=strmatch('ini',vars);
            if(~isempty(k))                
            %     svm=fitcsvm(trainData,trainClass(:,1));
            %     ini=svm.Beta;                
                obj.optimisationSettings.ini=vals{k};                
            else
                obj.optimisationSettings.ini=[];
            end  
            k=strmatch('discriminator',vars);
            if(~isempty(k))
                obj.optimisationSettings.discriminator=vals{k};
            else
                obj.optimisationSettings.discriminator='ANA';
            end
%             k=strmatch('ntop',vars);            
            if(~isempty(k))
                np=vals{k};
                np=min([vals{k},numDimensions]); 
                % Dimesionality reduction cannot be larger than original dimension
                obj.optimisationSettings.ntop=np;
                if(obj.optimisationSettings.ntop > 1 && ... % ANA only works on 1D
                        strcmp(obj.optimisationSettings.discriminator,'ANA'))
                    obj.optimisationSettings.discriminator='1NN';
                end                
            else
                obj.optimisationSettings.ntop=1;
            end  
            obj.optimisationSettings.strfitnessfct = 'myfit';            
        end
        
        function obj=optimise(obj,trainData,trainClass)
            trainClass = trainClass(:,1);
            [all_transformations,thrs,coefs]=getWeights(obj,trainData,trainClass);
            
            obj.optimisationOutcome.weights=all_transformations(1:size(trainData,2),:);

            if(strcmpi(obj.optimisationSettings.discriminator,'1NN'))
                obj.optimisationOutcome.discriminator.myKnn=...
                    fitcknn(obj.applyTransformation(trainData),...
                    vec2ind(trainClass')', 'NumNeighbors',1);
            end
            if(strcmpi(obj.optimisationSettings.discriminator,'ANA'))     
                obj.optimisationOutcome.discriminator.threshold=thrs;
                obj.optimisationOutcome.weights=coefs*obj.optimisationOutcome.weights;
            end
        end
        function [current_transformation,thrs,coefs]=getWeights(obj,trainData,trainClass)
            
            switch obj.optimisationSettings.optimizer
                case 'cmaes'
                    [current_transformation,thrs,coefs]=...
                        purecmaes(obj.optimisationSettings.strfitnessfct,...
                        trainData,trainClass,[],obj.optimisationSettings.l,...
                        obj.optimisationSettings.show,...
                        obj.optimisationSettings.ini);
                case 'pso'
                    [current_transformation,thrs,coefs]=pso(...
                        obj.optimisationSettings.strfitnessfct,trainData,...
                        trainClass,[],obj.optimisationSettings.show,obj.optimisationSettings.l);
                case 'es'
                    [current_transformation,thrs,coefs]=es(...
                        obj.optimisationSettings.strfitnessfct,trainData,...
                        trainClass,[],obj.optimisationSettings.show,...
                        obj.optimisationSettings.ini,obj.optimisationSettings.l);
                otherwise
                    [current_transformation,thrs,coefs]=es(...
                        obj.optimisationSettings.strfitnessfct,trainData,...
                        trainClass,[],obj.optimisationSettings.show,...
                        obj.optimisationSettings.ini,obj.optimisationSettings.l);            
            end
        end
                        
        function visualise(obj,data,classes)
            oData=data;
            w=obj.optimisationOutcome.weights;
            figure;
            xr=data*w;
            maxBin=0;
            for i=0:1
                hold on;
                h=histogram(xr(classes==i),50,'Normalization','probability');
                maxBin=max(maxBin,max(h.Values));
            end
            t=obj.optimisationOutcome.discriminator.threshold;                
            hold on;
            plot(t*ones(1,100),0:maxBin/100:maxBin-maxBin/100,'--','LineWidth',2);

            xlabel('$\hat{x}=s\vec{x}\omega^T$','Interpreter','latex');
            ylabel('Probability');
            legend({'Class -1','Class 1','Optimal-margin threshold'});      

            if(size(data,2)>2)
                return;
            end
            figure;
            for i=1:size(data,1)
                dataMap(i,:)=((oData(i,:)*w)/(norm(w)^2))*w;
            end

            gscatter([oData(:,1);dataMap(:,1)],[oData(:,2);dataMap(:,2)]...
                ,[classes;classes+2],'brbr','..xx',[10,10,10,10]);
            xlabel('Feature 1');
            ylabel('Feature 2');
            hold on;
            c0=mean(oData(classes(:,1)==0,:));
            c1=mean(oData(classes(:,1)==1,:));
            xl=xlim;
            yl=ylim;
            yy1=([xl(1),yl(1)]*w/(norm(w)^2))*w;
            yy2=([xl(2),yl(2)]*w/(norm(w)^2))*w;            
            plot([yy1(1),yy2(1)],[yy1(2),yy2(2)],'--','LineWidth',2);
            %             gscatter([c0(1),c1(1)],[c0(2),c1(2)],[],'k','.',[30]);
            %             cy0=([c0(1),c0(2)]*w/(norm(w)^2))*w;
            %             cy1=([c1(1),c1(2)]*w/(norm(w)^2))*w;            
            %             gscatter([cy0(1),cy1(1)],[cy0(2),cy1(2)],[],'g','.',[30]);

            if(strcmpi(discr.type,'Opt'))     
                t=discr.thr(2)*w;                
                hold on;
                gscatter(t(1),t(2),[],'k','.',[30]);
            end

            if(strcmpi(obj.optimisationSettings.discriminator,'ANA'))    
                t=obj.optimisationOutcome.discriminator.threshold*w;               
                hold on;
                gscatter(t(1),t(2),[],'k','.',[30]);
                
                c=obj.optimisationOutcome.discriminator.threshold*w;                                
                x1=max(data(:,1));
                y1=(-x1*w(1)+c)/w(2);
                x2=min(data(:,1));
                y2=(-x2*w(1)+c)/w(2);
                plot([x1,x2],[y1,y2],'k-','LineWidth',2);
            end

            legend({'Class -1','Class 1','Transformed class -1',...
                'Transformed class 1','Transformation',...
                'Optimal-margin threshold','Separator hyperplane'});
            
        end
        
        function estClasses=predict(obj,data)            
            estClasses=getClass(obj,data);
        end
        
        function estClasses=getClass(obj,data)            
            dataToTest=data*obj.optimisationOutcome.weights;
            if(strcmpi(obj.optimisationSettings.discriminator,'1NN'))
                estClasses=predict(obj.optimisationOutcome.discriminator.myKnn,dataToTest);
            end
            
            if(strcmpi(obj.optimisationSettings.discriminator,'ANA'))
                t=obj.optimisationOutcome.discriminator.threshold;
%                 c=obj.optimisationOutcome.discriminator.coef;
                estClasses=-(double(dataToTest-t>0)-1);
            end            
        end
        
        function transferredData=applyTransformation(obj,data)
            transferredData=feval(obj.optimisationSettings.kernel,...
                data*obj.optimisationOutcome.weights);
        end
    end
end