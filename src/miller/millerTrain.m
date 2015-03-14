function [net,u,v,P] = millerTrain(inputData, outputData)

%% Do PCA to reduce input space
u=mean(inputData,1);
v=std(inputData,[],1);
inputData=bsxfun(@minus,inputData,u);
inputData=bsxfun(@rdivide,inputData,v);
[p,c,a]=pca(inputData);
PCs=50;  %Keeping first 40 principal components
reducedInputData=c(:,1:PCs);
P=p(:,1:PCs)';

%% Train NN as regression problem
%If we train each possible output independently (which should only mean
%more hidden units):
[net,tr] = trainNNregression(reducedInputData,outputData);

%% Alt: Train NN as classif problem (not sure this is working well)
%[net,tr] = trainNNclassif(reducedInputData,outputData);

% y=net(reducedInputData');
% y1=net1(reducedInputData');
% figure 
% hold on
% plot(outputData(:,1))
% plot(y>.5)
% plot(outputData(:,2))
% plot(y1>.5)
% legend('Actual','Predicted')
% hold off
end

