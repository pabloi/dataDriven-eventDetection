subject = 6;
trial = 3;

[ref_l, ref_r] = getStanceFromEvents(roundedEventArray(1000:end, :, subject, trial), eventsToBeUsed);
[out_l, out_r] = getStanceFromEvents(roundedEventArrayFVA(1000:end, :, subject, trial), eventsToBeUsed);
[deviation, mistakes, details] = ss_metric(ref_l, out_l);

figure;
t = 1:length(ref_l);
stem(deviation(:, 1), deviation(:, 2), 'b-o');
hold on; plot(t, details.mistake_raw, 'm:', t, details.mistake_weighted, 'r-');