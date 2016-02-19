------------------------------------------------------------------------
--[[ VRClassRewardCaptioner ]] --
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassRewardCaptioner, nn.SelectTable(-1))
------------------------------------------------------------------------
local VRClassRewardCaptioner, parent = torch.class("nn.VRClassRewardCaptioner", "nn.Criterion")

function VRClassRewardCaptioner:__init(module, scale, criterion)
    parent.__init(self)
    self.module = module -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = { torch.Tensor() }
    self.reward = {} -- 5 is batch size
end

function VRClassRewardCaptioner:updateOutput(input, target)
    assert(torch.type(input) == 'table')
    
    local reward = 0
    for i = 1, #input do
	local _maxVal
        local _maxIdx
	_maxVal, _maxIdx = torch.max(input[i][1], 2)
	reward = reward + torch.eq(_maxIdx, target[i])[1][1]
    end
    reward = reward * self.scale
    self.output = -reward
    table.insert(self.reward,reward)

    if self.sizeAverage then
        self.output = self.output / #input
    end
    return self.output
end

function VRClassRewardCaptioner:updateGradInput(inputTable, target)
    local baseline = inputTable[1][2]:transpose(1,2)
    local batch_size = target:size(2)
    local timestep = #inputTable
    local hidden_size = inputTable[1][1]:size(2)
    --print ("hidden_size:", hidden_size)
    --print ("timestep:", timestep)
    --print ("batch_size", batch_size)
    local reward = torch.Tensor(batch_size) -- 1 x batch
    for i = 1, batch_size do 
	reward[i] = self.reward[i]
    end
    --print ("reward:", reward)
    self.vrReward = reward
    -- reduce variance of reward using baseline
    --print ("baseline:", baseline)
    self.vrReward:add(-1, baseline)
    --print ("vrReward:", self.vrReward)

    if self.sizeAverage then
        self.vrReward:div(batch_size)
    end
    --print ("average vrReward:", self.vrReward)
    -- broadcast reward to modules
    self.module:reinforce(self.vrReward)

    self.gradInput = {}
    for i = 1, timestep do 
	local gradInput_item  = {}
        gradInput_item[1] = torch.Tensor(batch_size, hidden_size):fill(0)
	gradInput_item[2] = self.criterion:backward(baseline, reward)
	table.insert(self.gradInput, gradInput_item)
    end
    self.reward = {} -- reset
    return self.gradInput
end

function VRClassRewardCaptioner:type(type)
    self._maxVal = nil
    self._maxIdx = nil
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
