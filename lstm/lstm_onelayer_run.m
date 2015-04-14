%% load data
% load('dataArrays.mat')

%% take many
mo = squeeze(motionArray(:, :, 1:7, :));
ev = squeeze(roundedEventArray(:, :, 1:7, :));
[mo_ts, ev_ts] = lstm_reshape(mo, ev, eventsToBeUsed);

%% take subject
% mo = squeeze(motionArray(:, :, 1, :));
% ev = squeeze(roundedEventArray(:, :, 1, :));
% [mo_ts, ev_ts] = lstm_reshape(mo, ev, eventsToBeUsed);

%% test with one trial from this subject
% mo2 = squeeze(motionArray(:, :, 1, 3));
% ev2 = squeeze(roundedEventArray(:, :, 1, 3));
% [mo_ts2, ev_ts2] = lstm_reshape(mo2, ev2, eventsToBeUsed);

%% create network
n = 10;
% delay = 0;
delay = 10;

nn = lstm_onelayer(n, 2);
nnd = adddelay(nn, delay);

%% prepare timeseries
[Xs,Xi,Ai,Ts] = preparets(nnd, mo_ts, ev_ts);
Ts = Ts(1:end-delay);

%% prepare network
nnd = configure(nnd, Xs, Ts);
nnd = init(nnd);

%% train network
nnd = train(nnd, Xs, Ts, Xi, Ai);

%% save and send email
filename = sprintf('LSTM %s n%d d%d.mat', ...
    datestr(now, 'yyyymmdd-HHMMSS'), n, delay);
save(filename, 'nnd');
sendmail('yinzhong@andrew.cmu.edu', 'LSTM complete', 'LSTM complete', {filename});

%% test
% [TXs, TXi, TAi, TTs] = preparets(nndc, mo_ts2, {}, ev_ts2);
% test_ret = nnd(TXs, TXi, TAi);
% test_ans = TTs;
% 
% test_ret_m = cell2mat(test_ret);
% test_ret_m = (test_ret_m>0)*2 - 1;
% test_ans_m = cell2mat(test_ans);
% 
% tt = 1:length(test_ret_m);
% plot(tt, test_ret_m, 'bx', tt, test_ans_m, 'r.');