function ret = peak1(x) 

n = length(x);

x0 = [+inf;x(1:n-1)];
x1 = [x(2:n);+inf];

ret = (x > x0) & (x > x1);

end