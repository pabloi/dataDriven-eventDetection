-- run by calling th test.lua -model modelFileName -datafile testDataFileName
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'hdf5'

-- local BatchLoader = require 'data.BatchLoader'
local BatchLoader = require 'BatchLoader'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
require 'Embedding'                     -- class name is Embedding (not namespaced)
local model_utils=require 'model_utils'


cmd:text()
cmd:text('Test a simple sequence labeling LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-model','results','contains just the protos table, and nothing else')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',false,'false to use max at each timestep, true to sample at each timestep')
cmd:option('-length',200,'number of characters to sample')
cmd:option('-datafile','set1_2.h5','filename of hdf5 data file')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)

local vocab_size = 3 -- 3 possible classes: class 0 is double swing (should never happen), single Stance L, single stance R, double stance


protos = torch.load(opt.model .. '.t7')
opt.rnn_size = protos.embed.weight:size(2)


-- load the testing data:
    local myFile = hdf5.open('../../data/' .. opt.datafile)
    local data = myFile:all()
out=torch.zeros(); --need to initialize with the proper size: T x nSubs x nTrials
for subjects, trials in pairs(data) do
     for trial, samples in pairs(trials) do
         local xData = samples['X']
         local yData = samples['y']


		-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
		local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size) -- initialize to zeros matrix
		local initstate_h = initstate_c:clone()


    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,xData:size(1) do
        -- embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
        embeddings[t] = xData;

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(protos.lstm:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

        predictions[t] = protos.softmax:forward(lstm_h[t])
		out[t][subject][trial]=predictions[t]:max();
    end
    loss = loss / opt.seq_length;
	torch.save(opt.datafile .. '_' .. opt.modelfile .. 'res.t7', out) -- temporary. Eventually we want to convert out to stance information and save it in .h5 too.

	end
end
