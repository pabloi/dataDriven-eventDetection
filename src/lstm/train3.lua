cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')

cmd:text('Data:')
cmd:option('-datafile','set1_1','filename of hdf5 data file')
cmd:option('-batch_size', 10, 'size of minibatch')
cmd:option('-seq_length', 500, 'number of timesteps to unroll to')
cmd:option('-seq_intro_length', 50, 'number of initial timesteps to exclude from loss')
cmd:option('-no_label_output', false, 'classify (stanceL, stanceR) directly instead of converting to multi-class')
cmd:text()

cmd:text('Training:')
cmd:option('-seed', 20150508, 'torch manual random number generator seed')
cmd:option('-max_epochs', 100, 'number of full passes through the training data')
cmd:option('-lr', 1e-1, 'learning rate')
cmd:option('-lrd', 1e-2, 'learning rate decay for adagrad,  in epochs: e.g .1 means it takes 10 epochs for the learning rate to decay to half its initial value,  20 epochs MORE to reach 1/4,  40 epochs MORE to 1/8 and so on')
cmd:option('-clamp', 1, 'gradient clamp')
cmd:option('-gpu', false, 'use GPU or not')
cmd:option('-gpu_ondemand', false, 'only load one mini-batch at a time to GPU')
cmd:text()

cmd:text('Network Size:')
cmd:option('-hidden_size', 100, 'size of hidden layer (before LSTM)')
cmd:option('-lstm_1_size', 40, 'number of cells in LSTM layer 1')
cmd:option('-lstm_2_size', 30, 'number of cells in LSTM layer 2')
cmd:text()

cmd:text('Network Parameters:')
cmd:option('-hidden_dropout', 0.2, 'hidden: dropout probability')
cmd:option('-lstm_peephole', false, 'LSTM: whether cell->gate feedback connections exist')
cmd:option('-lstm_squash', false, 'LSTM: whether cell output is squashed with tanh')
cmd:text()

cmd:text('I/O:')
cmd:option('-savefile', 'model_autosave', 'filename to autosave the model (net) to,  appended with the epoch in which it was saved.t7')
cmd:option('-save_every', 1, 'save every epoch') -- Changed to describe # of saving points as a function of epochs (instead of iterations)
cmd:option('-print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:text()

opt = cmd:parse(arg)

-- shorthands from opt
local seq_length = opt.seq_length
local seq_intro_length = opt.seq_intro_length
local seq_real_length = seq_length - seq_intro_length
local batch_size = opt.batch_size
local batch_samples = batch_size*seq_real_length

-- load corresponding torch & nn library
if opt.gpu then
    require 'init_GPU'
else
    require 'init_CPU'
end

local BatchLoader = require 'BatchLoader2'
local LSTM = require 'LSTM3'
local model_utils=require 'model_utils'
-- require 'mattorch'

function isnan(x) return x ~= x end

-- random seed
torch.manualSeed(opt.seed)

-- load data
--local loader = BatchLoader.create('../../data/' .. opt.datafile .. '.h5', batch_size, seq_length)
local loader = BatchLoader.create(opt.datafile, batch_size, seq_length)
local input_size = loader.input_size
local output_size = loader.output_size
opt.input_size = input_size
opt.output_size = output_size

-- build network
local net = {}

net.hidden = nn.Sequential():
    add(nn.Dropout(opt.hidden_dropout)):
    add(nn.Linear(opt.input_size, opt.hidden_size)):
    add(nn.ReLU(true))

net.lstm_1 = LSTM(opt.hidden_size, opt.lstm_1_size, opt)
net.lstm_2 = LSTM(opt.lstm_1_size, opt.lstm_2_size, opt)

if not opt.no_label_output then
    -- merged {stanceL, stanceR, stanceLR} label classification
    local weights = torch.Tensor{0.34, 0.34, 0.32}
    net.output = nn.Sequential():
        add(nn.Linear(opt.lstm_2_size, 3)):
        add(nn.LogSoftMax())
    net.criterion = nn.ClassNLLCriterion(weights)
else
    -- separate stanceL and stanceR classification
    -- Sigmoid + BCECriterion c.f. <http://qr.ae/0cUq0> (by Jack Rae, Google DeepMind)
    net.output = nn.Sequential():
        add(nn.Linear(opt.lstm_2_size, output_size)):
        add(nn.Sigmoid())
    net.criterion = nn.BCECriterion()
end

-- flatten parameters tensor
local params, grad_params = model_utils.combine_all_parameters(net.hidden, net.lstm_1, net.lstm_2, net.output)
params:uniform(-0.08, 0.08)

-- clone for backprop thru time (BPTT)
local net_clones = {}
for name, orig in pairs(net) do
    print('cloning '..name)
    net_clones[name] = model_utils.clone_many_times(GPU(orig), seq_length)
end

-- zeros for LSTM initial state and final gradient
local lstm_1_zero = GPU(torch.zeros(batch_size, opt.lstm_1_size))
local lstm_2_zero = GPU(torch.zeros(batch_size, opt.lstm_2_size))

-- optimization score function : BPTT
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
    local hidden = {} -- hidden layer output
    local lstm_1_c = {[0]=lstm_1_zero} -- LSTM 1 state
    local lstm_1_h = {[0]=lstm_1_zero} -- LSTM 1 output
    local lstm_2_c = {[0]=lstm_2_zero} -- LSTM 1 state
    local lstm_2_h = {[0]=lstm_2_zero} -- LSTM 1 output
    local output = {} -- output layer output
    local loss = 0

    for t = 1, seq_length do
        xt = x:select(1, t)
        hidden[t] = net_clones.hidden[t]:forward(xt)
        lstm_1_c[t], lstm_1_h[t] = unpack(net_clones.lstm_1[t]:forward{hidden[t]  , lstm_1_c[t-1], lstm_1_h[t-1]})
        lstm_2_c[t], lstm_2_h[t] = unpack(net_clones.lstm_2[t]:forward{lstm_1_h[t], lstm_2_c[t-1], lstm_2_h[t-1]})
        output[t] = net_clones.output[t]:forward(lstm_2_h[t])
        if t > seq_intro_length then
            yt = y:select(1, t)
            newloss = net_clones.criterion[t]:forward(output[t], yt)
            loss = loss + newloss
        end
    end
    loss = loss/batch_samples

    ------------------ backward pass -------------------
    -- loss gradient w.r.t. :
    local doutput                       -- output layer output
    local dlstm_2_h                     -- LSTM 2 output @ t
    local dlstm_2_h_last = lstm_2_zero  -- LSTM 2 output @ t+1
    local dlstm_2_c_last = lstm_2_zero  -- LSTM 2 state  @ t+1
    local dlstm_1_h                     -- LSTM 1 output @ t
    local dlstm_1_h_last = lstm_1_zero  -- LSTM 1 output @ t+1
    local dlstm_1_c_last = lstm_1_zero  -- LSTM 1 state  @ t+1
    local dhidden                       -- hidden layer output

    for t = seq_length, seq_intro_length+1, -1 do
        xt = x:select(1, t)
        yt = y:select(1, t)

        doutput = net_clones.criterion[t]:backward(output[t], yt)
        dlstm_2_h = net_clones.output[t]:backward(lstm_2_h[t], doutput)
        dlstm_1_h, dlstm_2_c_last, dlstm_2_h_last = unpack(net_clones.lstm_2[t]:backward(
            {lstm_1_h[t], lstm_2_c[t-1], lstm_2_h[t-1]},
            {dlstm_2_c_last, dlstm_2_h:add(dlstm_2_h_last)}
        ))
        dhidden  , dlstm_1_c_last, dlstm_1_h_last = unpack(net_clones.lstm_1[t]:backward(
            {hidden[t]  , lstm_1_c[t-1], lstm_1_h[t-1]},
            {dlstm_1_c_last, dlstm_1_h:add(dlstm_1_h_last)}
        ))
        _ = net_clones.hidden[t]:backward(xt, dhidden)
    end

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
