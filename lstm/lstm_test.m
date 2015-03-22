%% load data
% load('dataArrays.mat')

%% take subject 1
mo = squeeze(motionArray(:, :, 1, :));
ev = squeeze(roundedEventArray(:, :, 1, :));
[mo_ts, ev_ts] = lstm_reshape(mo, ev, eventsToBeUsed);

%% testing
mo2 = squeeze(motionArray(:, :, 1, 3));
ev2 = squeeze(roundedEventArray(:, :, 1, 3));
[mo_ts2, ev_ts2] = lstm_reshape(mo2, ev2, eventsToBeUsed);

%% prepare network and time series
delay = 0;
% delay = 10;

nn = lstm_naive;

[Xs,Xi,Ai,Ts] = preparets(nnd, mo_ts, {}, ev_ts);

if delay > 0
    nnd = adddelay(lstm_naive, delay);
    Ts = {Ts{1:end-delay+1}};
else
    nnd = nn;
end

%% train
nndt = train(nnd, Xs, Ts, Xi, Ai); % FIXME