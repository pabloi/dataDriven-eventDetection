-- adapted from: wojciechz/learning_to_execute on github

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm(opt)
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()--
        -- transforms input
        -- local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(x)
        local i2h            = nn.Linear(opt.input_size, opt.rnn_size)(x) --Linear transformation of the input (i2h=Mi*x+bi). Weights are stored internally (?) 
        -- transforms previous timestep's output
        local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h) --Linear transform, h2h=Mh*x + bh
        return nn.CAddTable()({i2h, h2h}) -- new_input = i2h + h2h;
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum()) 

    local next_c           = nn.CAddTable()({ -- c = forget_gate.*c + in_gate .* in_transform
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- CMultTable is element-wise multiplication (?). functionally this is h=tanh(c) .* out_gate

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h}) -- this defines a module with three inuputs: x,c,h and two outputs: c,h
end

return LSTM

