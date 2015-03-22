function plot_ts(ts, i, varargin)

r = cell2mat(ts);
plot(r(i, :), varargin{:});

end