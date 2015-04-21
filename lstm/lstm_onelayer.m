function net = lstm_onelayer(n, n_out)
% constructs uninitialized RNN with:
%   * One input timeseries `x` (arbitrary dimension)
%   * One meta-layer of `n` LSTM cells (close-loop feedback)
%   * One output layer
%   * One output timeseries `y` (dimension `n_out`)

net = network;
net.name = 'LSTM one-layer';

net.performFcn = 'crossentropy';
net.performParam.regularization = 0.1;

net.trainFcn = 'trainscg';

net.divideFcn = 'divideblock';
net.divideMode = 'value';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% dimensions and I/O
N = 8*n + 1;

net.numInputs = 1;
net.numLayers = N;

net.outputConnect(N) = 1;
net.outputs{N}.name = 'y';

net.inputs{1}.name = 'x';

% LSTM cells setup
L = {
    'i'  , 'netsum' , 'logsig'
    'f'  , 'netsum' , 'logsig'
    'o'  , 'netsum' , 'logsig'
    'z'  , 'netsum' , 'tansig'
    'iz' , 'netprod', 'purelin'
    'fc' , 'netprod', 'purelin'
    'h'  , 'netprod', 'tansig'
    'c'  , 'netsum' , 'purelin'
};
conn = [1 2 3 4 5 6
        5 6 7 5 8 8];
fdbk = [7 7 7 7 8 8 8 8 8
        1 2 3 4 1 2 3 6 7];
for i = 0:n-1
    j = 8*i;
    net.biasConnect(j+(1:4)) = 1; % bias -> i, f, o, z
    net.inputConnect(j+(1:4), 1) = 1; % x -> i, f, o, z
    for c = j+conn
        net.layerConnect(c(2), c(1)) = 1;
    end
    for c = j+fdbk
        net.layerConnect(c(2), c(1)) = 1;
        net.layerWeights{c(2), c(1)}.delays = 1;
    end
    net.layerConnect(N, j+7) = 1; % h -> output layer
    for k = 1:8
        net.layers{j+k}.name = sprintf('%s%d', L{k, 1}, i);
        net.layers{j+k}.netInputFcn = L{k, 2};
        net.layers{j+k}.transferFcn = L{k, 3};
        net.layers{j+k}.size = 1;
        net.layers{j+k}.initFcn = 'initnw';
    end
end

% output layer setup
net.layers{N}.name = 'out';
net.layers{N}.netInputFcn = 'netsum';
% net.layers{N}.transferFcn = 'purelin';
% net.layers{N}.transferFcn = 'hardlims';
net.layers{N}.transferFcn = 'logsig';
net.layers{N}.size = n_out;
net.layers{N}.initFcn = 'initnw';
net.biasConnect(N) = 1;

end