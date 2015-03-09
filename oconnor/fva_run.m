[t, ~, d1, d2] = size(motionArray);
eventArrayFVA = false(size(eventArray));
for i = 1:d1
    for j = 1:d2
        [LHS, RHS, LTO, RTO] = fva(squeeze(motionArray(:, :, i, j)));
        eventArrayFVA(:, :, i, j) = [LHS RTO RHS LTO];
    end
end
