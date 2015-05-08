-- adapted from: wojciechz/learning_to_execute on github
require 'nn'
require 'nngraph'
local LSTM = function (inputSize, outputSize, opt)
    -- input
    local x = nn.Identity()()
    -- feedback
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    -- create one {input/output/forget} gate with optional cell feedback connection
    function new_gate(has_c2g)
        -- linear transformation of the input : i2g=Mi*x+bi
        local i2g = nn.Linear(inputSize, outputSize)(x)
        -- linear transformation of output from previous timestep : h2g = Mh*prev_h + bh
        local h2g = nn.Linear(outputSize, outputSize)(prev_h)
        if has_c2g then
            local c2g = nn.CMul(outputSize)(prev_c)
            return nn.CAddTable(){i2g, h2g, c2g} -- new_gate = i2g + h2g + c2g
        else
            return nn.CAddTable(){i2g, h2g} -- new_gate = i2g + h2g
        end
    end

    local in_gate          = nn.Sigmoid()(new_gate(opt.lstm_peephole))
    local forget_gate      = nn.Sigmoid()(new_gate(opt.lstm_peephole))
    local out_gate         = nn.Sigmoid()(new_gate(opt.lstm_peephole))
    local in_transform     = nn.Tanh()(new_gate(false)) -- this never has lstm_peephole

    -- c = forget_gate.*c + in_gate .* in_transform
    local next_c = nn.CAddTable()({
        nn.CMulTable(){forget_gate, prev_c},
        nn.CMulTable(){in_gate, in_transform}
    })

    -- h=tanh(c) .* out_gate
    -- simplification by J. Schmidthuber 2002: remove tanh
    if opt.lstm_squash then
        next_c = nn.Tanh()(next_c)
    end
    local next_h = nn.CMulTable(){next_c, out_gate}

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end
return LSTM
