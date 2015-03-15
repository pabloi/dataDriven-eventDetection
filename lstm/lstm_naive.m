function nn = lstm_naive()

nn = network;
nn.name = 'LSTM naive';


% boundary

nn.numInputs = 1;  % x ; h', c'
nn.numLayers = 8;  % i, f, o, z, iz, fc', h, c
nn.outputConnect = [0 0 0 0 0 0 1 1]; % h, c

nn.outputs{7}.name = 'h';
nn.outputs{7}.feedbackDelay = 1;
nn.outputs{7}.feedbackInput = 2;
nn.outputs{7}.feedbackMode = 'open';

nn.outputs{8}.name = 'c';
nn.outputs{8}.feedbackDelay = 1;
nn.outputs{8}.feedbackInput = 3;
nn.outputs{8}.feedbackMode = 'open';

nn.inputs{1}.name = 'x';
nn.inputs{2}.name = 'h_';
nn.inputs{3}.name = 'c_';


% connection

nn.biasConnect = [1 1 1 1 0 0 0 0].'; % i, f, o, z

nn.inputConnect = [
    1 1 1 1 0 0 0 0 % x  -> i, f, o, z
    1 1 1 1 0 0 0 0 % h' -> i, f, o, z
    1 1 1 0 0 1 1 0 % c' -> i, f, o, fc', h
].';

LC = zeros(8);
LC(1, 5) = 1; % i   -> iz
LC(2, 6) = 1; % f   -> fc'
LC(3, 7) = 1; % o   -> h
LC(4, 5) = 1; % z   -> iz
LC(5, 8) = 1; % iz  -> c
LC(6, 8) = 1; % fc' -> c
nn.layerConnect = LC.';


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
    nn.layers{i}.name = L{i, 1};
    nn.layers{i}.netInputFcn = L{i, 2};
    nn.layers{i}.transferFcn = L{i, 3};
end