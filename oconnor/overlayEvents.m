function overlayEvents(t, e, y1, y2, varargin)

if nargin <= 4
    varargin = {'k:'};
end

hold on;
n = size(e, 1);
for i = 1:n
    if e(i)
        plot([t(i) t(i)], [y1 y2], varargin{:});
    end
end
hold off;

end