function fva_compare(motionArray, eventArray, eventArrayFVA, i, j, k)

m = motionArray(:, :, i, j);
e = eventArray(:, k, i, j);
ef = eventArrayFVA(:, k, i, j);

tt = size(m, 1);

plotFCV(m);
overlayEvents(1:tt, e, -3, 3, 'b:');
overlayEvents(1:tt, ef, -3, -2.5, 'b-');