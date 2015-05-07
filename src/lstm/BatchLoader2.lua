require 'torch'
require 'math'
require 'hdf5'

local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_file, batch_size, seq_length)
    local self = {}
    setmetatable(self, BatchLoader)

    self.batch_size = batch_size
    self.seq_length = seq_length

    print(string.format(
        'loading data file "%s" into batches of %d seqs * %d timesteps',
        data_file, batch_size, seq_length))

    local myFile = hdf5.open(data_file, 'r')
    local subjects = myFile:all()
    myFile:close()

    --[[
        format of subjects:
        {subject_id = {trial_id = {
            X = torch.Tensor(timesteps, input_size),
            y = torch.Tensor(timesteps, output_size)
        }}}
    --]]

    -- chop into sequences and:
    --  1. save in each trial
    --  2. count total # of sequences
    local ns = 0
    local input_size = 0
    local output_size = 0
    for _, trials in pairs(subjects) do
        for _, trial in pairs(trials) do
            local x = trial.X ; trial.X = nil
            local y = trial.y ; trial.y = nil
            if ns == 0 then
                input_size = x:size(2)
                output_size = y:size(2)
            end
            local len = x:size(1)
            local rem = len % seq_length
            if rem ~= 0 then
                len = len - rem
                x = x:narrow(1, 1, len)
                y = y:narrow(1, 1, len)
            end
            local n = len / seq_length
            trial.x = x:view(seq_length, n, input_size)
            trial.y = y:view(seq_length, n, output_size)
            trial.n = n
            ns = ns + n
        end
    end
    -- collect into single tensor of all sequences
    xs = torch.Tensor(seq_length, ns, input_size)
    ys = torch.Tensor(seq_length, ns, output_size)
    local i = 1
    for _, trials in pairs(subjects) do
        for _, trial in pairs(trials) do
            local n = trial.n
            xs:narrow(2, i, n):copy(trial.x)
            ys:narrow(2, i, n):copy(trial.y)
            i = i + n
        end
    end

    if not opt.no_label_output then
        ys = ys:select(3, 1) + ys:select(3, 2)*2
    end

    self.xs = xs
    self.ys = ys
    self.ns = ns
	self.input_size = input_size
    self.output_size = output_size

    self.evaluated_batches = 0 -- number of times next_batch() called

    print('data loaded')
    subjects = nil
    collectgarbage()
    return self
end

function BatchLoader:next_batch()
    local nb = self.batch_size
    local i = math.random(1, self.ns - nb + 1)
    --print('next batch starting from :', i)
    return self.xs:narrow(2, i, nb),
           self.ys:narrow(2, i, nb)
end

return BatchLoader
