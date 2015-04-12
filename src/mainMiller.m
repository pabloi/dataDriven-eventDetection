%% Test Miller NN scheme
clear all
close all
%% Load some data & get classes
load ../data/dataArrays_stroke.mat

%% get Training data
motionArray(isnan(motionArray))=0; %Needed to replace NaN entries
fs=1000;
features=zeros(size(motionArray,1),42,size(motionArray,3),size(motionArray,4));
stanceL=zeros(size(motionArray,1),1,size(motionArray,3),size(motionArray,4));
stanceR=zeros(size(motionArray,1),1,size(motionArray,3),size(motionArray,4));
for j=1:size(motionArray,3)
    for k=1:size(motionArray,4)
        [features(:,:,j,k), labels] = getNNFeatures(motionArray(:,:,j,k),markersToBeUsed,fs,10);
        [stanceL(:,1,j,k),stanceR(:,1,j,k)] = getStanceFromEvents(eventArray(:,:,j,k),eventsToBeUsed);
    end
end

outputs=cat(2,stanceL,stanceR); %Using stance as output, instead of events themselves
[inputData,outputData,T] = millerSetup(features,outputs,fs); %Separates data into sequences for training, as Miller does

%% TrainNN
[net,u,v,P] = millerTrain(inputData, outputData);

%% Test NN
[output] = millerTest(features,net,u,v,P,T);
%         figure 
%         hold on
%         plot(outputs(:,1,1,1))
%         plot(output(1,:))
%         plot(outputs(:,2,1,1))
%         plot(output(2,:))
%         legend('Actual 1','Predicted 1', 'Actual 2', 'Predicted 2')
%         hold off
%% Compute events from stance
outputEventArray=false(size(eventArray));
for j=1:size(motionArray,3)
    for k=1:size(motionArray,4)
        [LHS,RHS,LTO,RTO] =getEventsFromStance(output(1,:,j,k)'>.5,output(2,:,j,k)'>.5); %Getting events from stances, to compare performance
        for i=1:length(eventsToBeUsed)
            eval(['aux=' eventsToBeUsed{i} ';'])
            outputEventArray(:,i,j,k)=aux;
        end
    end
end

%%
[ trueErrors ] = computeTrueErrorsRobust(eventArray,outputEventArray,200);
clear h
f=figure;
for j=1:size(motionArray,3)
for k=1:size(motionArray,4)
subplot(size(motionArray,3),size(motionArray,4),sub2ind([size(motionArray,3),size(motionArray,4)],j,k))
for i=1:length(eventsToBeUsed)
    h(i,:)=hist(trueErrors{i,j,k},[-30:1:30]);
end
bar([-30:1:30],h')
legend('LHS','RTO','RHS','LTO')
ylabel(['Sub ' num2str(j) ', trial ' num2str(k) ])
end
end
saveFig(f,'./','MillerResultsStroke');

%% Save
save MillerResults_stroke.mat net u v P outputEventArray

