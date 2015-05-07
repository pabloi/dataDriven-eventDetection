-- adapted from: wojciechz/learning_to_execute on github

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm(opt)
    local input_size = opt.input_size
    local rnn_size = opt.rnn_size

    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    -- create one {input/output/forget} gate
    function new_gate()
        -- linear transformation of the input : i2h=Mi*x+bi
        local i2h = nn.Linear(input_size, rnn_size)(x)
        -- linear transformation of output from previous timestep : h2h = Mh*prev_h + bh
        local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
        return nn.CAddTable()({i2h, h2h}) -- new_gate = i2h + h2h
    end

    local in_gate          = nn.Sigmoid()(new_gate())
    local forget_gate      = nn.Sigmoid()(new_gate())
    local out_gate         = nn.Sigmoid()(new_gate())
    local in_transform     = nn.Tanh()(new_gate()) 

    -- c = forget_gate.*c + in_gate .* in_transform
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })

    -- h=tanh(c) .* out_gate
    local next_h           = nn.CMulTable()({nn.Tanh()(next_c), out_gate}) 

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h}) -- this defines a module with three inuputs: x,c,h and two outputs: c,h
end

return LSTM
