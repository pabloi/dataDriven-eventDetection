% Load miller
load('dataArrays_upSampled.mat')
[goldL, goldR] = getStanceFromEvents(eventArray, eventsToBeUsed);

load('MillerResults_upSampled.mat')
out_events = outputEventArray;
[outL, outR] = getStanceFromEvents(out_events, eventsToBeUsed);

% TODO workaround (Ts is a dummy)
Ts = 1; 

[deviationL, mistakeL, d_listL] = ss_metric(goldL, outL);
[deviationR, mistakeR, d_listR] = ss_metric(goldR, outR);
