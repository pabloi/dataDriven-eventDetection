%% load data
load('dataArrays.mat')

%% take subject 1
mo = squeeze(motionArray(:, :, 1, :));
ev = squeeze(roundedEventArray(:, :, 1, :));
[mo_ts, ev_ts] = lstm_reshape(mo, ev, eventsToBeUsed);

%% prepare network and time series
delay = 20;

nn = lstm_naive;
nnd = adddelay(lstm_naive, delay);

[Xs,Xi,Ai,Ts] = preparets(nnd, mo_ts, {}, ev_ts);
Ts = {Ts{1:end-delay+1}};

%% train
nndt = train(nnd, Xs, Ts, Xi, Ai); % FIXME