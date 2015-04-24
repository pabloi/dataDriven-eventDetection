-- run by calling th test.lua -model modelFileName -datafile testDataFileName
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'mattorch'

local LSTM = require 'LSTM'             -- LSTM timestep and utilities
require 'Embedding'                     -- class name is Embedding (not namespaced)
local model_utils=require 'model_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a simple sequence labeling LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-model','results','contains just the protos table, and nothing else')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',false,'false to use max at each timestep, true to sample at each timestep')
cmd:option('-length',200,'number of characters to sample')
cmd:option('-datafile','set1_2','filename of hdf5 data file')
cmd:option('-rnn_size',54,'size of LSTM internal state') -- This should be read from the loaded model
cmd:option('-batch_size',1,'number of sequences to train on in parallel') --Is this necessary?
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)

local vocab_size = 3 -- 3 possible classes: class 0 is double swing (should never happen), single Stance L, single stance R, double stance


protos = torch.load('./trainedModels/' .. opt.model .. '.t7')


-- load the testing data:
loaded = mattorch.load('../../data/' .. opt.datafile .. '.mat');
xData=loaded['X']; -- data from all trials
yData=loaded['y'];
xData=xData:transpose(1,3); -- for some reason mattorch returns the dimensions in the inverse order
yData=yData:transpose(1,3);
aux=torch.zeros(xData:size(1),xData:size(3),vocab_size); --need to initialize with the proper size: T x nSubs x nTrials
out=torch.zeros(xData:size(1),2,xData:size(3));
for i=1,xData:size(3) do

	partialData=xData:select(3,i); --Extracting slice (subtensor) along dimension 3, just index i 
	partialLabels=yData:select(3,i);

		-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
		local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size) -- initialize to zeros matrix
		local initstate_h = initstate_c:clone()


    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,partialData:size(1) do
        -- embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
        embeddings[t] = partialData:transpose(1,2):select(2,t); --transpose does the obvious, select slices along dimension 2, keeping index t only
        --print('embeddings[t]:dim()=' ..embeddings[t]:dim())
        --print('embeddings[t]:size(1)=' ..embeddings[t]:size(1))

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(protos.lstm:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = protos.softmax:forward(lstm_h[t])
		loss = loss + protos.criterion:forward(predictions[t], partialLabels[t][1]+2*partialLabels[t][2])
		for k=1,vocab_size do
			aux[t][i][k]=predictions[t][k]; --log probability of sample t belonging to class k
		end
		-- HERE WE NEED TO FIND THE MAX PROB CLASS AND ASSIGN 'out' CORRESPONDINGLY
		--if aux[t][i]==1 then
		--	out[t][1][i]=1;
		--end
		--if aux[t][i]==2 then
		--	out[t][2][i]=1;
		--end
		--if aux[t][i]==3 then
		--	out[t][1][i]=1;
		--	out[t][2][i]=1;
		--end
    end
    loss = loss / partialData:size(1);
	print(string.format('Finished trial %4d, loss is %6.8f',i,loss))
end
	--torch.save(opt.datafile .. '_' .. opt.model .. 'res.t7', out) -- temporary. Eventually we want to convert out to stance information and save it in .h5 too.
list = {p = aux:transpose(1,3)};
mattorch.save('../../data/lstmResults/lstm_Test' .. opt.datafile .. '_Train' .. opt.model .. '.mat',list)
