%% Test Miller NN scheme
function [predictedYTest,predictedYTrain,net,u,v,P,T]=mainMiller(XTrain,yTrain,XTest,markersToBeUsed,threshold)
%Assume XTrain is TxNxP markers
%Assume yTrain is TxMxP labels

if nargin<5 || isempty(threshold)
    threshold=.5; %Default thresholding value
end

%% get the actual Training data by performing feature extraction
motionArray=XTrain;
fs=100;
features=zeros(size(motionArray,1),42,size(motionArray,3),size(motionArray,4));
for j=1:size(motionArray,3)
    for k=1:size(motionArray,4) %This is 1
        [features(:,:,j,k), labels] = getNNFeatures(motionArray(:,:,j,k),markersToBeUsed,fs,10);
    end
end

outputs=yTrain; %Using stance as output, instead of events themselves
[inputData,outputData,T] = millerSetup(features,outputs,fs); %Separates data into sequences for training, as Miller does

%% TrainNN
[net,u,v,P] = millerTrain(inputData, outputData);
%If we want to check performance on training set:
[output] = millerTest(features,net,u,v,P,T);
predictedYTrain=permute(output>threshold,[2,1,3]); %Thresholding to get stance

%% Test NN with the test set
% Get the actual testing data by feature extraction
motionArray=XTest;
features=zeros(size(motionArray,1),42,size(motionArray,3),size(motionArray,4));
for j=1:size(motionArray,3)
    for k=1:size(motionArray,4) %This is 1
        [features(:,:,j,k), labels] = getNNFeatures(motionArray(:,:,j,k),markersToBeUsed,fs,10);
    end
end

%Test:
[output] = millerTest(features,net,u,v,P,T);
predictedYTest=permute(output>threshold,[2,1,3]); %Thresholding to get stance

