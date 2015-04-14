function [mo_ts, ev_ts] = lstm_reshape(mo, ev, evLabels, delay)

if nargin <= 3
    delay = 0;
end

sz = size(mo);
T = sz(1); % total time (in samples)
F = sz(2); % # of markers
R = prod(sz(3:end)); % # of runs

% merge all runs and convert motion array to time series format
mor = reshape(mo, T, F, R);
mop = permute(mor, [2 3 1]);
mo_ts = squeeze(mat2cell(mop, F, R, ones(1, T))).';

% shift range of stance-swing into $\pm 1$ and pad with delay
function y = prep(x)
    y = x*2 - 1;
    if delay > 0
        y = [ones(1, delay)*y(1) y(1:end-delay).'];
    else
        y = y.';
    end
end

ev2 = zeros(2, R, T);
for i = 1:R
    [l, r] = getStanceFromEvents(ev(:, :, i), evLabels);
    ev2(1, i, :) = prep(l);
    ev2(2, i, :) = prep(r);
end
ev_ts = squeeze(mat2cell(ev2, 2, R, ones(1, T))).';

end