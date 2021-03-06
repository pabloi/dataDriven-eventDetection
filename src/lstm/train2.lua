cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-datafile','set1_1','filename of hdf5 data file')
cmd:option('-batch_size',10,'size of minibatch')
cmd:option('-seq_length',750,'number of timesteps to unroll to')
cmd:option('-seq_intro_length',100,'number of initial timesteps to exclude from loss')
cmd:option('-rnn_size',100,'size of LSTM internal state')
cmd:option('-max_epochs',100,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (net) to, appended with the epoch in which it was saved.t7')
cmd:option('-save_every',1,'save every epoch') -- Changed to describe # of saving points as a function of epochs (instead of iterations)
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',20150507,'torch manual random number generator seed')
cmd:option('-lr',5e-2,'learning rate')
cmd:option('-lrd',1e-2,'learning rate decay for adagrad, in epochs: e.g .1 means it takes 10 epochs for the learning rate to decay to half its initial value, 20 epochs MORE to reach 1/4, 40 epochs MORE to 1/8 and so on')
cmd:option('-clamp',20,'gradient clamp')
cmd:option('-gpu',false,'use GPU or not')
cmd:option('-gpu_ondemand',false,'only load one mini-batch at a time to GPU')
cmd:option('-no_label_output',false,'do NOT transform output (stanceL, stanceR) to single class label')
cmd:option('-lstm_cell_feedback',false,'whether cell->gate feedback connections exist in LSTM cells')
cmd:text()
opt = cmd:parse(arg)

-- shorthands from opt
local seq_length = opt.seq_length
local seq_intro_length = opt.seq_intro_length
local seq_real_length = seq_length - seq_intro_length
local batch_size = opt.batch_size
local batch_samples = batch_size*seq_real_length

-- CPU or GPU
if opt.gpu then
    require 'init_GPU'
else
    require 'init_CPU'
end

local BatchLoader = require 'BatchLoader2'
local LSTM = require 'LSTM'
local model_utils=require 'model_utils'
-- require 'mattorch'
-- require 'BinaryCriterion'

function isnan(x) return x ~= x end 

-- random seed
torch.manualSeed(opt.seed)

-- load data
-- local loader = BatchLoader.create('../../data/' .. opt.datafile .. '.h5', batch_size, seq_length)
local loader = BatchLoader.create(opt.datafile, batch_size, seq_length)
local input_size = loader.input_size
local output_size = loader.output_size
opt.input_size = input_size
opt.output_size = output_size

-- build network
local net = {}
local criterion = nil

net.lstm = GPU(LSTM.lstm(opt))
if not opt.no_label_output then
    -- merged {stanceL, stanceR, stanceLR} label classification
    local weights = torch.Tensor{0.34, 0.34, 0.32}
    net.output = GPU(nn.Sequential():add(nn.Linear(opt.rnn_size, 3)):add(nn.LogSoftMax()))
    criterion = GPU(nn.ClassNLLCriterion(weights))
else
    -- separate stanceL and stanceR classification
    -- Sigmoid() + BCECriterion() c.f. <http://qr.ae/0cUq0> (by Jack Rae, Google DeepMind)
    net.output = GPU(nn.Sequential():add(nn.Linear(opt.rnn_size, output_size)):add(nn.Sigmoid()))
    criterion = GPU(nn.BCECriterion())
end

-- flatten parameters tensor
local params, grad_params = model_utils.combine_all_parameters(net.lstm, net.output)
params:uniform(-0.08, 0.08)

-- clone (expand in time)
local net_clones = {}
for name, orig in pairs(net) do
    print('cloning '..name)
    net_clones[name] = model_utils.clone_many_times(orig, seq_length)
end

-- LSTM initial state/output => zero
local initstate_c = GPU(torch.zeros(batch_size, opt.rnn_size))
local initstate_h = initstate_c:clone()
-- LSTM final state's backward message (dloss/dfinalstate) => zero
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

-- optimization score function
local current_batch = 1
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    -- dims: {time, batch, input/output_size}
    -- if label_output: 3rd dim of y is collapsed
    local x, y = loader:next_batch()
    if opt.gpu_ondemand then
        x = GPU(x)
        y = GPU(y)
    end
    current_batch = current_batch + 1

    ------------------- forward pass -------------------
    local lstm_c = {[0]=initstate_c} -- LSTM state
    local lstm_h = {[0]=initstate_h} -- LSTM output
    local output = {} -- network output
    local loss = 0

    for t=1,seq_length do
        xt = x:select(1, t)
        yt = y:select(1, t)
        lstm_c[t], lstm_h[t] = unpack(net_clones.lstm[t]:forward{xt, lstm_c[t-1], lstm_h[t-1]})
        output[t] = net_clones.output[t]:forward(lstm_h[t])
        newloss = criterion:forward(output[t], yt)
        if t > seq_intro_length then
            loss = loss + newloss
        end
    end
    loss = loss/batch_samples

    ------------------ backward pass -------------------
    -- loss gradient w.r.t. :
    local dlstm_h_last = dfinalstate_h -- LSTM output
    local dlstm_c_last = dfinalstate_c -- LSTM state
    for t=seq_length,seq_intro_length+1,-1 do
        xt = x:select(1, t)
        yt = y:select(1, t)

        -- backprop through criterion, output
        local doutput_t = criterion:backward(output[t], yt)
        local dlstm_h_t = dlstm_h_last + net_clones.output[t]:backward(lstm_h[t], doutput_t)

        -- backprop through LSTM
        _, dlstm_c_last, dlstm_h_last = unpack(net_clones.lstm[t]:backward(
            {xt, lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c_last, dlstm_h_t}
        ))
    end

    --[[
    initstate_h:copy(lstm_h[seq_length])
    initstate_c:copy(lstm_c[seq_length])
    --]]

    -- scale then clamp
    grad_params:div(batch_samples):clamp(-opt.clamp, opt.clamp)

    return loss, grad_params
end

-- optimization
torch.save(string.format(opt.savefile .. '_Params.t7', epoch) , opt)
local optim_state = {learningRate = opt.lr, learningRateDecay = opt.lrd/loader.ns} -- For no decay set learningRateDecay=0, decay is implemented as inversely proportional to number of epoch evals in this way.
local iterations = opt.max_epochs * loader.ns/batch_size --loader.ns is the size of each epoch in iterations
local losses = torch.zeros(iterations)
for i = 1, iterations do -- one iteration is going through just 1 chunk of sequence, of length seq_length. If we have 30 sequences of 25secs each, with seq_length=100 (1s) it takes 25*30 =750 iterations to go through all the data once 
	local epoch = i*batch_size/loader.ns -- not an integer usually
    local _, loss = optim.adagrad(feval, params, optim_state)
    loss = loss[1]
    losses[i] = loss

    if (i*batch_size) % (opt.save_every*loader.ns) == 0 then 
        torch.save(string.format( opt.savefile .. '_Epoch%4d.t7', epoch), net)
		torch.save(string.format( opt.savefile .. '_Loss.t7', epoch), opt)
    end
    if (i*batch_size) % (opt.print_every*loader.ns) == 0 then
        print(string.format("epoch %4d, loss = %6.8f, gradnorm = %6.4e", epoch, loss, grad_params:norm()))
    end
	if i == 1 then -- forcing initial print & save
        print(string.format("epoch %4d, loss = %6.8f, gradnorm = %6.4e", epoch, loss, grad_params:norm()))
		torch.save(string.format( opt.savefile .. '_Epoch%4d.t7', epoch), net)
    end
	if i == iterations then -- final save: get losses into matlab which is the easy thing to do
		opt.loss = losses
        if mattorch then
            mattorch.save(opt.savefile .. '_Results.mat', losses)
        end
	end
end
