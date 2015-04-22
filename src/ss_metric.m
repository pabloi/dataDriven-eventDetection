function [deviation, mistake, d_list] = ss_metric(Ts, ref, out, p)
% Ts : sample time
% ref, out : {stance=0, swing=1} column vector

if nargin < 4
    p = struct ...
        ( 'width', 0.20 ...
        , 'bounce', 3 ...
        );
    
end

N = length(ref);

% deviations are measured at events (transitions)
d_ref = [0; diff(ref)];
ev_ref = [1; find(d_ref ~= 0); N];
mask = ones(N, 1);
d_list = zeros(length(ev_ref) - 2, 1);
d_n = 0;

function cover(t, width, d)
    d_n = d_n + 1;
    d_list(d_n) = d;
    mask((t-width):(t+width)) = 0;
end

for i = 2:length(ev_ref)-1
    t = ev_ref(i); % time of event
    dir = d_ref(t); % positive or negative transition
    dt = min(t - ev_ref(i-1), ev_ref(i+1) - t); % distance to closest event
    width = dt * p.width; % half width of symmetric window
    slice = out((t-width):(t+width)) * dir; % content within window (sign-corrected)
    
    % wrong signs at beginning or end => mistake
    if slice(1) ~= 0 || slice(end) ~= 1, continue; end

    d_slice = diff(slice);
    ev_slice = find(d_slice ~= 0);
    if isscalar(ev_slice) == 1
        % single transition
        cover(t, width, ev_slice - t);
    end
    
    
end

end