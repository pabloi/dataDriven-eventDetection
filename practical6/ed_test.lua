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


cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
-- cmd:option('-vocabfile','vocabfile.t7','filename of the string->int table')
cmd:option('-datafile','data.h5','filename of hdf5 data file')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',100,'number of timesteps to unroll to')
cmd:option('-rnn_size',54,'size of LSTM internal state')
cmd:option('-max_epochs',500,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file') -- This needs to be at least larger than max_epochs * nBatches so that it saves at least once.
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
    {save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
    .. '.t7'

local vocab_size = 5 -- 5 possible classes: no event, LHS, RHS, LTO, RTO


protos = torch.load(opt.model)
opt.rnn_size = protos.embed.weight:size(2)

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size) -- initialize to zeros matrix
local initstate_h = initstate_c:clone()

-- do fwd

    -- TODO rewrite
	aux = torch.zeros(opt.seq_length);
	for t=1,opt.seq_length do -- very inefficient way to assign class labels instead of what we have now.
		for j=1,4 do
			if y[j][t]==1 then
				aux[t]=j;
			end
		end
		if aux[t]==0 then
			aux[t]=5;
		end
	end

    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,opt.seq_length do
        -- embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
        embeddings[t] = x[{{}, t}]

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(protos.lstm:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

        predictions[t] = protos.softmax:forward(lstm_h[t])

        print('predictions[t]:type()=' ..predictions[t]:type())
        print('predictions[t]:dim()=' ..predictions[t]:dim())
        print('predictions[t]:size(1)=' ..predictions[t]:size(1))
        print('y[{{}, t}]:size(1)=' ..y[{{}, t}]:size(1))
        print('predictions[t][1]=' ..predictions[t][1])
		print('predictions[t][2]=' ..predictions[t][2])
		print('predictions[t][3]=' ..predictions[t][3])
		print('predictions[t][4]=' ..predictions[t][4])
		print('predictions[t][5]=' ..predictions[t][5])
        print('label[t]=' ..aux[t])

        loss = loss + protos.criterion:forward(predictions[t], aux[t])
    end
    loss = loss / opt.seq_length
