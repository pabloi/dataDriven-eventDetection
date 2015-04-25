[t, ~, d1] = size(X);
y_out = zeros(size(y));
for i = 1:d1
    [LHS, RHS, LTO, RTO] = fva(squeeze(X(:, :, i)));
    [L, R] = getStanceFromEvents([LHS, RHS, LTO, RTO], {'LHS', 'RHS', 'LTO', 'RTO'});
    y_out(:, :, i) = [L, R];
end