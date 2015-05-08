%Summarize results
%This is to plot results from ./test.sh and ./test2.sh from LSTM, and
%compare to Miller & O'Connor
%% set1_2 with 1_1 train
clearvars
train='1_1';
test='1_2';
for i=[10:20:50]
    interpretLSTMoutput(['BEST_LSTM' num2str(i) '.mat'])
end
for i=[10:20:50]
    load(['BEST_LSTM' num2str(i) '.mat'])
    eval(['predictedY' num2str(i) '=predictedY;']);
end
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/millerResults/miller_Testset' test '_Trainset' train '_100.mat'])
predictedYtest100=predictedYtest;
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/millerResults/miller_Testset' test '_Trainset' train '_33.mat'])
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/oconnorResults/oconnor_Testset' test '.mat'])
load(['../set' test '.mat'])
names={'LSTM 10 epochs','Miller 33','Miller 100','O''Connor','LSTM 30 epochs','LSTM 50 epochs'};
compareResults(y,names,predictedY10,predictedYtest,predictedYtest100,y_out,predictedY30,predictedY50)
set(gcf,'Units','Normalized','OuterPosition',[0,0,1,1])
saveFig(gcf,'../',['Test' test 'Train' train 'Compare'])

%% set1_1 with 1_2 train
clearvars
train='1_2';
test='1_1';
for i=[10:20:50]
    interpretLSTMoutput(['BEST_LSTM' num2str(i) '_2.mat'])
end
for i=[10:20:50]
    load(['BEST_LSTM' num2str(i) '_2.mat'])
    eval(['predictedY' num2str(i) '=predictedY;']);
end
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/millerResults/miller_Testset' test '_Trainset' train '_33.mat'])
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/oconnorResults/oconnor_Testset' test '.mat'])
load(['../set' test '.mat'])
names={'LSTM 10 epochs','Miller 33','O''Connor','LSTM 30 epochs','LSTM 50 epochs'};
compareResults(y,names,predictedY10,predictedYtest,y_out,predictedY30,predictedY50)
set(gcf,'Units','Normalized','OuterPosition',[0,0,1,1])
saveFig(gcf,'../',['Test' test 'Train' train 'Compare'])

%% set2_2 with 2_1 train
clearvars
train='2_1';
test='2_2';
for i=[10:20:50]
    interpretLSTMoutput(['BEST2_LSTM' num2str(i) '.mat'])
end
for i=[10:20:50]
    load(['BEST2_LSTM' num2str(i) '.mat'])
    eval(['predictedY' num2str(i) '=predictedY;']);
end
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/millerResults/miller_Testset' test '_Trainset' train '_33.mat'])
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/oconnorResults/oconnor_Testset' test '.mat'])
load(['../set' test '.mat'])
names={'LSTM 10 epochs','Miller 33','O''Connor','LSTM 30 epochs','LSTM 50 epochs'};
compareResults(y,names,predictedY10,predictedYtest,y_out,predictedY30,predictedY50)
set(gcf,'Units','Normalized','OuterPosition',[0,0,1,1])
saveFig(gcf,'../',['Test' test 'Train' train 'Compare'])

%% set2_1 with 2_2 train -- not yet done
clearvars
train='2_2';
test='2_1';
for i=[10:20:50]
    interpretLSTMoutput(['BEST2_LSTM' num2str(i) '_2.mat'])
end
for i=[10:20:50]
    load(['BEST2_LSTM' num2str(i) '_2.mat'])
    eval(['predictedY' num2str(i) '=predictedY;']);
end
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/millerResults/miller_Testset' test '_Trainset' train '_33.mat'])
load(['/Datos/Documentos/PhD/asig/MachineLearning/project/data/oconnorResults/oconnor_Testset' test '.mat'])
load(['../set' test '.mat'])
names={'LSTM 10 epochs','Miller 33','O''Connor','LSTM 30 epochs','LSTM 50 epochs'};
compareResults(y,names,predictedY10,predictedYtest,y_out,predictedY30,predictedY50)
set(gcf,'Units','Normalized','OuterPosition',[0,0,1,1])
saveFig(gcf,'../',['Test' test 'Train' train 'Compare'])