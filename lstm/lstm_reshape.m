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

% pre-process events into stance-swing ($\pm 1$)
% FIXME: right leg is discarded
ev2 = zeros(1, R, T);
for i = 1:R
    [l, ~] = getStanceFromEvents(ev(:, :, i), evLabels);
    l = l*2 - 1;
    if delay > 0
        l = [ones(1, delay)*l(1) l(1:end-delay).'];
    else
        l = l.';
    end
    ev2(1, i, :) = l;
end
ev_ts = squeeze(mat2cell(ev2, 1, R, ones(1, T))).';

end