#!/usr/bin/env lua
-- loader for character-level language models

require 'torch'
require 'math'
require 'hdf5'

local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_file, subject, trial, batch_size, seq_length)
    local self = {}
    setmetatable(self, BatchLoader)

    -- construct a tensor with the data for the subject and trial
    print('loading data files...')
    local myFile = hdf5.open(data_file)
    local pathToData = '/' .. subject .. '/' .. trial
    local data = myFile:read(pathToData):all()
    local xData = data['X']
    local yData = data['y']

    -- X11 = data['1']['1']['X'][{{1,opt.seq_length}, {}}]
    -- y11 = data['1']['1']['y'][{{1,opt.seq_length}, {}}]
    -- TODO where to transpose
    -- local x = X11:t()
    -- local y = y11:t()

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        xData = xData:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
        yData = yData:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    -- self.x_batches = xData:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.x_batches = xData:split(batch_size*seq_length, 1)
    self.nbatches = #self.x_batches
    -- self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.y_batches = yData:split(batch_size*seq_length, 1)
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
    return self.x_batches[self.current_batch], self.y_batches[self.current_batch]
end

return BatchLoader
