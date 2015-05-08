function y = normalize1(x)

y = (x - mean(x))./std(x);

end