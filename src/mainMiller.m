%% Test Miller NN scheme

%% Load some data & get classes
load ../data/dataArrays_upSampled.mat

%% get Training data
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
[output] = millerTest(features(:,:,1,1),net,u,v,P,T);
        figure 
        hold on
        plot(outputs(:,1,1,1))
        plot(output(1,:))
        plot(outputs(:,2,1,1))
        plot(output(2,:))
        legend('Actual 1','Predicted 1', 'Actual 2', 'Predicted 2')
        hold off
%% Compute events from stance
[LHS,RHS,LTO,RTO] =getEventsFromStance(output(1,:)'>.5,output(2,:)'>.5); %Getting events from stances, to compare performance

%%
[ trueErrors ] = computeTrueErrorsRobust(eventArray(:,:,1,1),[LHS,RTO,RHS,LTO],200);
for i=1:length(trueErrors)
    h(i,:)=hist(trueErrors{i},[-30:1:30]);
end
figure
bar([-30:1:30],h')
legend('LHS','RTO','RHS','LTO')
