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

if delay > 0
    nnd = adddelay(lstm_naive, delay);
    [Xs,Xi,Ai,Ts] = preparets(nnd, mo_ts, {}, ev_ts);
    Ts = Ts(1:end-delay+1);
else
    nnd = nn;
    [Xs,Xi,Ai,Ts] = preparets(nnd, mo_ts, {}, ev_ts);
end

nnd = configure(nnd, Xs, Ts);
nnd = init(nnd);

%% train
nnd = train(nnd, Xs, Ts, Xi, Ai); % FIXME

%% test
nndc = closeloop(nnd);
[TXs, TXi, TAi, TTs] = preparets(nndc, mo_ts2, {}, ev_ts2);
test_ret = nndc(TXs, TXi, TAi);
test_ans = TTs;

test_ret_m = cell2mat(test_ret);
test_ret_m = (test_ret_m>0)*2 - 1;
test_ans_m = cell2mat(test_ans);

tt = 1:length(test_ret_m);
plot(tt, test_ret_m, 'bx', tt, test_ans_m, 'r.');