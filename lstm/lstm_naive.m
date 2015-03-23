function net = lstm_naive()

net = network;
net.name = 'LSTM naive';


% boundary

net.numInputs = 1;  % x ; h', c'
net.numLayers = 8;  % i, f, o, z, iz, fc', h, c
net.outputConnect = [0 0 0 0 0 0 1 0]; % h, c

net.outputs{7}.name = 'h';
net.outputs{7}.feedbackDelay = 1;
net.outputs{7}.feedbackInput = 2;
net.outputs{7}.feedbackMode = 'open';

% nn.outputs{8}.name = 'c';
% nn.outputs{8}.feedbackDelay = 1;
% nn.outputs{8}.feedbackInput = 3;
% nn.outputs{8}.feedbackMode = 'open';

net.inputs{1}.name = 'x';
net.inputs{2}.name = 'h_';
% nn.inputs{3}.name = 'c_';


% connection

net.biasConnect = [1 1 1 1 0 0 0 0].'; % i, f, o, z

net.inputConnect = [
    1 1 1 1 0 0 0 0 % x  -> i, f, o, z
    1 1 1 1 0 0 0 0 % h' -> i, f, o, z
].';

LC = zeros(8);
LC(1, 5) = 1; % i   -> iz
LC(2, 6) = 1; % f   -> fc'
LC(3, 7) = 1; % o   -> h
LC(4, 5) = 1; % z   -> iz
LC(5, 8) = 1; % iz  -> c
LC(6, 8) = 1; % fc' -> c
LC(8, :) = [1 1 1 0 0 1 1 0]; % c' -> i, f, o, fc', h (internal feedback)
net.layerConnect = LC.';

% setup internal feedback delay
for i = find(LC(8, :))
    net.layerWeights{i, 8}.delays = 1;
end


% layers

L = {
    'i'  , 'netsum' , 'logsig'
    'f'  , 'netsum' , 'logsig'
    'o'  , 'netsum' , 'logsig'
    'z'  , 'netsum' , 'tansig'
    'iz' , 'netprod', 'purelin'
    'fc_', 'netprod', 'purelin'
    'h'  , 'netprod', 'tansig'
    'c'  , 'netsum' , 'purelin'
};
for i = 1:8
    net.layers{i}.name = L{i, 1};
    net.layers{i}.netInputFcn = L{i, 2};
    net.layers{i}.transferFcn = L{i, 3};
    net.layers{i}.size = 1;
    net.layers{i}.initFcn = 'initnw';
end

% nn = closeloop(nn);
net.trainFcn = 'trainlm';
net.divideFcn = 'dividerand';
net.divideMode = 'value';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
