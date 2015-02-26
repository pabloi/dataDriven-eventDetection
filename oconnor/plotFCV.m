function plotFCV(fc)

n = size(fc, 1);
t = 1:n;

ly = fc(:, 2);
lz = fc(:, 3);
ry = fc(:, 5);
rz = fc(:, 6);

lzv = [0;diff(lz)];
rzv = [0;diff(rz)];

plot(t, normalize1(ly), 'b-', ...
     t, normalize1(lz), 'r-', ...
     t, normalize1(lzv), 'm-');

end