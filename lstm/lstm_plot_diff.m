%%
subject = 4;
trial = 1;

%%
mo2 = squeeze(motionArray(:, :, subject, trial));
ev2 = squeeze(roundedEventArray(:, :, subject, trial));
[mo_ts2, ev_ts2] = lstm_reshape(mo2, ev2, eventsToBeUsed);

[Xs2, Xi2, Ai2, Ts2] = preparets(nnd, mo_ts2, ev_ts2);
test_ret = nnd(Xs2, Xi2, Ai2);
test_ans = Ts2(1:end-delay);

test_ret_m = cell2mat(test_ret);
test_ret_m = (test_ret_m>0)*2 - 1;
test_ans_m = cell2mat(test_ans);

tt = 1:length(test_ret_m);

%%

hold off; plot(tt, test_ans_m(1, :)*2, 'k:');
hold on; stem(tt, (test_ret_m(1, :) - test_ans_m(1, :))*1.1, 'b-');