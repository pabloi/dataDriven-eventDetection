%run Miller for some dataset
function runMiller(trainFilename,testFilename)
%trainFilename='set1_1'; %Data to load and name to use for saving
%testFilename='set1_2'; %Data to load and name to use for saving
%% Load
load(['../../data/' trainFilename '.mat']); 
X1=X;
y1=y;
load(['../../data/' testFilename '.mat']); 
X2=X;
y2=y;

%% Miller run
[predictedYtest,predictedYtrain]=mainMiller(X1,y1,X2,markerLabels,.5); %Using .5 as threshold, and X1,y1 as training set, testing over X2

%% Save
save(['../../data/millerResults/miller_Test' testFilename '_Train' trainFilename '.mat']);

end
%% Check training:
% figure
% subplot(1,2,1)
% hold on
% plot(reshape(y1(:,1,:),numel(y1)/2,1))
% plot(reshape(predictedY1_train(:,1,:),numel(y1)/2,1))
% legend('y1','predicted y1 when training on X1,y1')
% hold off
% subplot(1,2,2)
% hold on
% plot(reshape(y2(:,1,:),numel(y2)/2,1))
% plot(reshape(predictedY2(:,1,:),numel(y2)/2,1))
% legend('y2','predicted y2 when training on X1,y1',)
% hold off
% 
% %% Quantify:
% 
% sum(abs(double(y1(:))-double(predictedY1_train(:))))/numel(y1)
% sum(abs(double(y1(:))-double(predictedY1(:))))/numel(y1)
% 
% sum(abs(double(y2(:))-double(predictedY2_train(:))))/numel(y2)
% sum(abs(double(y2(:))-double(predictedY2(:))))/numel(y2)
