------------------------------------------------------------------------
--[[ VRCIDErReward ]] --
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRCIDErReward, nn.SelectTable(-1))
------------------------------------------------------------------------

require '../misc/cider_scorer'
local utils = require '../misc/utils'

local VRCIDErReward, parent = torch.class("nn.VRCIDErReward", "nn.Criterion")

function VRCIDErReward:__init(module, scale, vocab)
    parent.__init(self)
    self.module = module -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = nn.MSECriterion() -- baseline criterion --#TODO: check whether should be MSECriterion
    self.sizeAverage = true
    self.gradInput = { torch.Tensor() }
    self.reward = {}
    self.vocab = vocab -- vocab is ix_to_word
    self.CiderScorer = CiderScorer()
end

function VRCIDErReward:int2word(index)
-- given the index of word, return the word in vocab table
    for k, v in pairs(self.vocab) do
	if k == index then
	    return v
	end
    end
end

function VRCIDErReward:updateOutput(input, target)
    
    --print ("VRCIDErReward:updateOutput")
    --print (input)
    --print ("up is input")
    --print (target)
    --print ("up is target")
    --io.read(1)

    assert(torch.type(input) == 'table')

    self.batch_size = target:size(1)
    self.nStep = target:size(2) 
    local inputs = utils.reformat(input, self.batch_size, self.nStep)
    local rewards = 0

    for i = 1, self.batch_size do
	local sample = inputs[i]
	local generated_sentence = ""
	local ground_truth_sentence = ""
	for t = 1, self.nStep do
	    -- ground_truth_sentence.append(word)
	    local ref_word_idx_t = target[i][t]
	    if ref_word_idx_t == 0 then -- if padding 0
		break -- also skip the generated, because we don't want to account for the 0 paddings
	    end
	    local ref_word_t = self.vocab[tostring(ref_word_idx_t)]
	    ground_truth_sentence = ref_word_t .. " " .. ground_truth_sentence 

	    -- generated_sentence.append(word)
	    local gen_word_prob_t = sample[t] -- generated word probobility at timestep t
	    local _, gen_word_t = torch.max(gen_word_prob_t, 1)
	    gen_word_t = self.vocab[tostring(gen_word_t[1])] --unpack and get the word by index
	    generated_sentence = gen_word_t .. " " .. generated_sentence --append word, insert space in between
	end
	print ("generated_sentence")
	print (generated_sentence)
	io.read(1)
	print ("ground_truth_sentence")
	print (ground_truth_sentence)
	io.read(1)
    end

    debug = false

    local reward = 0
    for i = 1, #input do
        local _maxVal
        local _maxIdx
        _maxVal, _maxIdx = torch.max(input[i][1], 2)
        reward = reward + torch.eq(_maxIdx, target[i])[1][1]
    end
    reward = reward * self.scale
    --self.output = -reward
    self.output = reward
    table.insert(self.reward,reward)

    if self.sizeAverage then
        self.output = self.output / #input
    end

    if debug then 
	self.output = 0
	print ("VRClassReward:", self.output)
    	return self.output
    end

    print ("VRClassReward:", self.output)
    return self.output
end

function VRCIDErReward:updateGradInput(inputTable, target)
    local baseline = inputTable[1][2]:transpose(1,2)
    local batch_size = target:size(2)
    local timestep = #inputTable
    local hidden_size = inputTable[1][1]:size(2)
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

    if debug then 
	for k,v in pairs(self.gradInput) do 
	    v[1]:fill(0)
	    v[2]:fill(0)
	end

    	return self.gradInput
    end
    return self.gradInput
    
end

function VRCIDErReward:type(type)
    self._maxVal = nil
    self._maxIdx = nil
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
