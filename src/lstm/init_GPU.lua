require 'torch'
require 'cutorch'
print(cutorch.getDeviceProperties(cutorch.getDevice()))
require 'cunn'
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    if cudaComputeCapability >= 3.5 then
        LookupTable = nn.LookupTableGPU
    else
        LookupTable = nn.LookupTable
    end
require 'nngraph'
require 'optim'
require 'hdf5'
function GPU(x)
  return x:cuda()
end
