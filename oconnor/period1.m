function ret = period1(x, k)

n = length(x);
t = 1:n;

xp = peak1(x);
tp = t(xp);
sdtp = sort(diff(tp));

ret = median(sdtp);

% if nargin < 2
%     k = 5;
% end
% np = length(tp);
% np_off = ceil(np/k);
% if np_off*2 > n
%     np_off = floor(n/2);
% end
% ret = mean(sdtp(np_off:np-np_off));


end