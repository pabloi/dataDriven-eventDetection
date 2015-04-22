% Load groundtruth
load('dataArrays_upSampled.mat')
[goldL, goldR] = getStanceFromEvents(eventArray, eventsToBeUsed);

% Load Miller
load('MillerResults_upSampled.mat')
out_events = outputEventArray;
[outL, outR] = getStanceFromEvents(out_events, eventsToBeUsed);

% Load O'Connor
% load('oconnerResults.mat')
% out_events = outputEventArray;
% [outL, outR] = getStanceFromEvents(out_events, eventsToBeUsed);

[deviationL, mistakeL, d_listL] = ss_metric(goldL, outL);
[deviationR, mistakeR, d_listR] = ss_metric(goldR, outR);

agg_dev = [deviationL; deviationR];
agg_dev = agg_dev(:,2);
agg_mistake = (mistakeL + mistakeR) / 2;

res = [mean(abs(agg_dev)), std(agg_dev), agg_mistake];
outpath = 'results_miller.txt';
dlmwrite(outpath,res,'\t')
hist(agg_dev);
xlabel('dt')
saveas(gcf, 'figures/hist_miller.eps', 'epsc')
