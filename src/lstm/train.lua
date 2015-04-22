require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'hdf5'
--require 'BatchLoader'

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
cmd:option('-datafile','../../data/set1_1.h5','filename of hdf5 data file')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',1000,'number of timesteps to unroll to')
cmd:option('-rnn_size',54,'size of LSTM internal state')
cmd:option('-max_epochs',50,'number of full passes through the training data')
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

local loader = BatchLoader.create(opt.datafile, opt.batch_size, opt.seq_length)
-- local vocab_size = loader.vocab_size  -- the number of distinct characters

local vocab_size = 4

-- define model prototypes for ONE timestep, then clone them
--
protos = {} -- TODO: local
-- protos.embed = Embedding(vocab_size, opt.rnn_size)
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.lstm = LSTM.lstm(opt)
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
local weights=torch.Tensor(5);
weights[1]=0.4;
weights[2]=0.4;
weights[3]=0.2;
protos.criterion = nn.ClassNLLCriterion(weights)

-- put the above things into one flattened parameters tensor
-- local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.softmax)
local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
clones = {} -- TODO: local
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size) -- initialize to zeros matrix
local initstate_h = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

local current_batch = 1
-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()
    --print('Load batch #' .. current_batch)
    current_batch = current_batch + 1
    -- print('x:size()=(' ..x:size(1) ..',' ..x:size(2) ..')')
    -- print('y:size()=(' ..y:size(1) ..',' ..y:size(2) ..')')


    -- assigning labels
	aux = torch.zeros(opt.seq_length);
for t=1,opt.seq_length do
	aux[t]= y[1][t] + 2*y[2][t];
end -- Assuming the y variable loaded is a 2xT tensor, with its first row being stanceL and second stanceR. Assigns label=1 to stanceL, label=2 to stanceR, label=3 to double stance, label=0 to both feet off the air (impossible!).

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
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

        predictions[t] = clones.softmax[t]:forward(lstm_h[t])

        --print('predictions[t]:type()=' ..predictions[t]:type())
        --print('predictions[t]:dim()=' ..predictions[t]:dim())
        --print('predictions[t]:size(1)=' ..predictions[t]:size(1))
        --print('y[{{}, t}]:size(1)=' ..y[{{}, t}]:size(1))
        --print('predictions[t][1]=' ..predictions[t][1])
		--print('predictions[t][2]=' ..predictions[t][2])
		--print('predictions[t][3]=' ..predictions[t][3])
        --print('label[t]=' ..aux[t])

        loss = loss + clones.criterion[t]:forward(predictions[t], aux[t])
		--print('loss=' ..loss)
		--loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dembeddings = {}                              -- d loss / d input embeddings
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], aux[t])
        dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)

        -- backprop through LSTM timestep
        dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through embeddings
        -- clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- optimization stuff
losses = {} -- TODO: local
local optim_state = {learningRate = 1e-1}
local iterations = opt.max_epochs * loader.nbatches
for i = 1, iterations do -- one iteration is going through just 1 chunk of sequence, of length seq_length. If we have 74 sequences of 50secs each, with seq_length=100 it takes 74*50 =~4000 iterations to go through all the data once 
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
    end
end
