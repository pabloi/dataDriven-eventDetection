% Load groundtruth
% load('dataArrays_upSampled.mat')
load('set1_1.mat')
% [goldL, goldR] = getStanceFromEvents(eventArray, eventsToBeUsed);
goldL = squeeze(y(:,1,:));
goldR = squeeze(y(:,2,:));

% Load Miller
% load('MillerResults_upSampled.mat')
% out_events = outputEventArray;
% [outL, outR] = getStanceFromEvents(out_events, eventsToBeUsed);

% Load O'Connor
load('oconnorResults/oconnor_Testset1_1.mat')
out_events = y_out;
outL = squeeze(y_out(:,1,:));
outR = squeeze(y_out(:,2,:));

% Load LSTM
% load('lstmResults/lstm_Testset1_1_Trainset1_1.h5327000train.mat')
% out_events = outputEventArray;
% [outL, outR] = getStanceFromEvents(out_events, eventsToBeUsed);

[deviationL, mistakeL, d_listL] = ss_metric(goldL, outL);
[deviationR, mistakeR, d_listR] = ss_metric(goldR, outR);

agg_dev = [deviationL; deviationR];
agg_dev = agg_dev(:,2);
agg_mistake = (mistakeL + mistakeR) / 2;

res = [mean(abs(agg_dev)), std(agg_dev), agg_mistake];
% outpath = 'results_miller.txt';
outpath = 'results_oconner_testset1_1.txt';
dlmwrite(outpath,res,'\t')
hist(agg_dev);
xlabel('dt')
saveas(gcf, 'figures/hist_oconner_testset1_1.eps', 'epsc')
