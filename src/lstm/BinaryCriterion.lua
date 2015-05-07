-- adapted from `nn.BCECriterion`
-- combines `nn.Sigmoid` and `nn.BCECriterion`
-- input range: [-inf, +inf]
-- BCE(sigmoid(x), t) = log(1+exp((1-t*2)*x)) where t = 0 or 1
local BinaryCriterion, parent = torch.class('nn.BinaryCriterion', 'nn.Criterion')

local eps = 1e-12

function BinaryCriterion:__init()
    parent.__init(self)
end

function BinaryCriterion:updateOutput(input, target)
    -- sign = (1-t*2)
    -- esx = exp(sign*x)
    -- out = log1p(esx) / n

    self.term1 = self.term1 or input.new()
    self.term2 = self.term2 or input.new()
    self.term3 = self.term3 or input.new()

    self.term1:resizeAs(input)
    self.term2:resizeAs(input)
    self.term3:resizeAs(input)

    local sign = self.term1
    local esx = self.term2
    local out = self.term3

    sign:fill(1):add(-2, target)
    esx:copy(input):cmul(sign):exp()
    out:copy(esx):log1p():div(target:nElement())

    return out:sum()
end

function BinaryCriterion:updateGradInput(input, target)
    -- sign = (1-t*2)
    -- esx = exp(sign*x) -- this is just reuse
    -- out = esx/(1+esx)/n

    self.term1 = self.term1 or input.new()
    self.term2 = self.term2 or input.new()
    self.term3 = self.term3 or input.new()

    self.term1:resizeAs(input)
    self.term2:resizeAs(input)
    self.term3:resizeAs(input)

    local sign = self.term1
    local esx = self.term2
    local esxp1 = self.term3

    sign:fill(1):add(-2, target)
    esx:copy(input):cmul(sign):exp()
    esxp1:copy(esx):add(1)

    self.gradInput:resizeAs(input)
    self.gradInput:copy(esx):cdiv(esxp1):cmul(sign):div(target:nElement())

    return self.gradInput
end
