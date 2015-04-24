function [deviation, mistake, details] = ss_metric(ref, out, p)
% ref, out : {swing=0, stance=1} column vector
% deviation : 2 columns (t, dt), each row describeing a matched event between `ref` and `out`
%   t : time of event in `ref`
%   dt : time of event in `out` - time of event in `ref`
% mistake : scalar indicator of severity of gross mistakes in prediction
%   higher number => more mistakes

if nargin < 3
    p = struct ...
        ( 'width', 0.20 ... width of window as percentage of distance to closest event
        , 'bounce', 2 ... penalty factor for multiple transitions within a window
        , 'gamma', 0.5 ... saturation exponent for extra penalty on consecutive mistakes
        , 'cooldown', 3 ... how fast consecutive mistakes are forgotten
        );
end

N = length(ref);

% deviations:
%   considered within symmetric window around events (transitions) in `ref`
%   successful matches exclude the window from mistake

d_ref = [0; diff(ref)];
ev_ref = [1; find(d_ref ~= 0); N];
n_ev_ref = length(ev_ref);
mask = ones(N, 1);

% pre-allocate deviation list
deviation = zeros(n_ev_ref - 2, 2);
dev_n = 0;

function push(t, width, d)
    dev_n = dev_n + 1;
    deviation(dev_n, :) = [t, d];
    mask((t-width):(t+width)) = 0;
end

for i = 2:(n_ev_ref-1)
    t = ev_ref(i); % time of event
    dir = d_ref(t); % positive or negative transition
    dt = min(t - ev_ref(i-1), ev_ref(i+1) - t); % distance to closest event
    width = ceil(dt * p.width); % half width of symmetric window
    
    % content within window (polarity corrected to simplify conditions)
    slice = out((t-width):(t+width))*dir + (dir == -1);
    
    % slice should start with 0 and end with 1; if not => mistake
    if slice(1) ~= 0 || slice(end) ~= 1, continue; end
    
    % calculate deviation
    d_slice = [0; diff(slice)];
    ev_slice = find(d_slice ~= 0);
    c_slice = width + 1; % center of slice == time of event
    if isscalar(ev_slice) == 1
        % single matching transition
        push(t, width, ev_slice - c_slice);
    else
        % multiple "bounce" transition => max deviation * penalty
        [~, ii] = max(abs(ev_slice - c_slice));
        m = ev_slice(ii) - c_slice;
        n = length(ev_slice);
        push(t, width, m*n*p.bounce);
    end
end
deviation = deviation(1:dev_n, :);

% mistakes:
%   penalize long periods of mistakes more than transients

mistake_raw = xor(ref, out) & mask; % raw mistake vector
mistake_weighted = zeros(N, 1);
cooldown = p.cooldown;
w = 0;
for i = 1:N
    if mistake_raw(i) == 1
        w = w + 1;
        mistake_weighted(i) = w;
    else
        w = max(w - cooldown, 0);
    end
end
mistake = sum((mistake_weighted.^p.gamma).*mistake_raw);

details = struct ...
    ( 'mistake_raw', mistake_raw ...
    , 'mistake_weighted', mistake_weighted ...
    );

end