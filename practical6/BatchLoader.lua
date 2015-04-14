#!/usr/bin/env lua
-- loader for character-level language models

require 'torch'
require 'math'
require 'hdf5'

local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_file, batch_size, seq_length)
    local self = {}
    setmetatable(self, BatchLoader)

    -- construct a tensor with the data for the subject and trial
    print('loading data files...')
    local myFile = hdf5.open(data_file)
    local data = myFile:all()

    -- self.batches is a table of tensors
    self.batch_size = batch_size
    self.seq_length = seq_length

    self.x_batches = {}
    self.nbatches = 0

    -- ipairs does not work because the indices are strings ('1') and not integers
    for subjects, trials in pairs(data) do
        for trial, samples in pairs(trials) do
            local xData = sample['X']
            local yData = sample['y']
            -- cut off the end so that it divides evenly
            local len = data:size(1)
            if len % (batch_size * seq_length) ~= 0 then
                xData = xData:sub(1, batch_size * seq_length
                            * math.floor(len / (batch_size * seq_length)))
                yData = yData:sub(1, batch_size * seq_length
                            * math.floor(len / (batch_size * seq_length)))
            end

            new_x_batches = xData:split(batch_size*seq_length, 1)
            self.y_batches = yData:split(batch_size*seq_length, 1)
            new_nbatches = #new_x_batches
            for k, v in ipairs(new_x_batches) do
                self.x_batches[#self.x_batches + 1] = v
            end
            for k, v in ipairs(new_y_batches) do
                self.y_batches[#self.y_batches + 1] = v
            end
            self.nbatches = self.nbatches + new_nbatches
        end
    end

    -- local pathToData = '/' .. subject .. '/' .. trial
    -- local data = myFile:read(pathToData):all()
    -- local xData = data['X']
    -- local yData = data['y']

    -- cut off the end so that it divides evenly
    -- local len = data:size(1)
    -- if len % (batch_size * seq_length) ~= 0 then
    --     xData = xData:sub(1, batch_size * seq_length
    --                 * math.floor(len / (batch_size * seq_length)))
    --     yData = yData:sub(1, batch_size * seq_length
    --                 * math.floor(len / (batch_size * seq_length)))
    -- end

    -- self.batches is a table of tensors
    -- self.batch_size = batch_size
    -- self.seq_length = seq_length

    -- self.x_batches = xData:view(batch_size, -1):split(seq_length, 2)
    -- self.x_batches = xData:split(batch_size*seq_length, 1)
    -- self.nbatches = #self.x_batches
    -- self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
    -- self.y_batches = yData:split(batch_size*seq_length, 1)
    assert(#self.x_batches == #self.y_batches)

    self.current_batch = 0
    self.evaluated_batches = 0  -- number of times next_batch() called

    print('data load done.')
    collectgarbage()
    return self
end

function BatchLoader:next_batch()
    self.current_batch = (self.current_batch % self.nbatches) + 1
    self.evaluated_batches = self.evaluated_batches + 1
    -- TODO transpose?
    return self.x_batches[self.current_batch]:t(),
        self.y_batches[self.current_batch]:t()
end

return BatchLoader
