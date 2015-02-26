function [LHS, RHS, LTO, RTO] = fva(motion)

[lfc, rfc, lt, rt, lh, rh] = motionToFeetCenter(motion);

n = size(motion, 1);
t = 1:n;

lfcy = lfc(:, 2);
lfcz = lfc(:, 3);
rfcy = rfc(:, 2);
rfcz = rfc(:, 3);

lfczv = [0;diff(lfcz)];
rfczv = [0;diff(rfcz)];

lfcyp = peak1(lfcy); 
rfcyp = peak1(rfcy);
lfczvp = peak1(lfczv);
lfczvpn = peak1(-lfczv);
rfczvp = peak1(rfczv);
rfczvpn = peak1(-rfczv);

lfcypt = find(lfcyp);
rfcypt = find(rfcyp);

t0 = period1(lfcy);
window = floor(t0/8);

LTO = false(n, 1);
for i = lfcypt.'
    lo = max(i - window, 1);
    hi = min(i + window, n);
    [~, j] = max(lfczv(lo:hi));
    LTO(j+lo-1) = true;
end
RTO = false(n, 1);
for i = rfcypt.'
    lo = max(i - window, 1);
    hi = min(i + window, n);
    [~, j] = max(rfczv(lo:hi));
    RTO(j+lo-1) = true;
end

LHS = false(n, 1);
RHS = false(n, 1);



end