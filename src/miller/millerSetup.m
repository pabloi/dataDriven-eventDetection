function [inputData,outputData,T] = millerSetup(features,outputs,fs)
%This function does the embedding to train the  Miller NN as a static NN
%(parses data into time-windows)

%% parse timeseries into sequences of 200ms, with the label being the one from the mid-sample
%Random pick M time-samples (per subject & trial) that have at least 200ms preceding and 200ms
%after
M=100;
T=round(.2*fs); %200ms in samples
i=randi(size(features,1)-2*T,M,1)+T;
inputData=zeros(length(i),size(features,3),size(features,4),2*T+1,size(features,2));
outputData=zeros(length(i),size(features,3),size(features,4),size(outputs,2));
for j=1:length(i)
    for l=1:size(features,4)
        for k=1:size(features,3)
            inputData(j,k,l,:,:)=features(i(j)-T:i(j)+T,:,k,l);
            outputData(j,k,l,:)=outputs(i(j),:,k,l);
        end
    end
end

s=size(inputData);
inputData=reshape(inputData,prod(s(1:3)),prod(s(4:5)));
outputData=reshape(outputData,prod(s(1:3)),size(outputs,2));

end

