function [LHS, RHS, LTO, RTO] = fva(motion)

[lfc, rfc, lt, rt, lh, rh] = motionToFeetCenter(motion);

n = size(motion, 1);
t = 1:n;

lhz = lh(:, 3);
rhz = rh(:, 3);

lfcy = lfc(:, 2);
lfcz = lfc(:, 3);
rfcy = rfc(:, 2);
rfcz = rfc(:, 3);

lfczv = [0;diff(lfcz)];
rfczv = [0;diff(rfcz)];

lfcypp = peak1(lfcy); 
rfcypp = peak1(rfcy);
lfcypn = peak1(-lfcy);
rfcypn = peak1(-rfcy);
lfcyppt = find(lfcypp);
rfcyppt = find(rfcypp);
lfcypnt = find(lfcypn);
rfcypnt = find(rfcypn);

% lfczvpn = peak1(-lfczv);
% rfczvpn = peak1(-rfczv);

t0 = period1(lfcy);
window_TO = floor(t0/8);

LTO = false(n, 1);
for i = lfcyppt.'
    lo = max(i - window_TO, 1);
    hi = min(i + window_TO, n);
    [~, j] = max(lfczv(lo:hi));
    LTO(j+lo-1) = true;
end
RTO = false(n, 1);
for i = rfcyppt.'
    lo = max(i - window_TO, 1);
    hi = min(i + window_TO, n);
    [~, j] = max(rfczv(lo:hi));
    RTO(j+lo-1) = true;
end

window_HS = floor(t0/8);
thres_HS = 0.35;

lhz_max = max(lhz);
lhz_min = min(lhz);
rhz_max = max(rhz);
rhz_min = min(rhz);
lhz_thres = (1-thres_HS)*lhz_min + thres_HS*lhz_max;
rhz_thres = (1-thres_HS)*rhz_min + thres_HS*rhz_max;

lfczv(lhz > lhz_thres) = NaN;
rfczv(rhz > rhz_thres) = NaN;

LHS = false(n, 1);
for i = lfcypnt.'
    lo = max(i - window_HS, 1);
    hi = min(i + window_HS, n);
    [~, j] = max(lfczv(lo:i));
    lo = lo + j - 1;
    [~, j] = min(lfczv(lo:hi));
    j = j + lo - 1;
    LHS(j) = true;
end
RHS = false(n, 1);
for i = rfcypnt.'
    lo = max(i - window_HS, 1);
    hi = min(i + window_HS, n);
    [~, j] = max(rfczv(lo:i));
    lo = lo + j - 1;
    [~, j] = min(rfczv(lo:hi));
    j = j + lo - 1;
    RHS(j) = true;
end



end