function plotFCV(motion)

[lfc, rfc, lt, rt, lh, rh] = motionToFeetCenter(motion);

n = size(motion, 1);
t = 1:n;

lfcy = lfc(:, 2);
lfcz = lfc(:, 3);
rfcy = rfc(:, 2);
rfcz = rfc(:, 3);

lfczv = [0;diff(lfcz)];
rfczv = [0;diff(rfcz)];

plot(t, normalize1(lfcy), 'b-' ...
    ,t, normalize1(lfcz), 'r-' ...
    ,t, normalize1(lfczv), 'm-' ...
    ,t, normalize1(motion(:, 3+4*9)), 'c-' ...
    );

legend('FCy', 'FCz', 'FCzv', 'HEEz');

end