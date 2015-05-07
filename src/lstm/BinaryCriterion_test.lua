require 'nn'
require 'BinaryCriterion'

x = torch.range(-10, 10, 2)
t = torch.Tensor{0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}

s = nn.Sigmoid()
bce = nn.BCECriterion()
bc = nn.BinaryCriterion()

sx = s:forward(x)
l_bce = bce:forward(sx, t)
g_bce = s:backward(x, bce:backward(sx, t))

l_bc = bc:forward(x, t)
g_bc = bc:backward(x, t)

print(l_bce, l_bc, l_bce - l_bc)
print((g_bce - g_bc):abs():max())
