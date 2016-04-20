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
    assert(torch.type(input) == 'table')

    self.batch_size = target:size(1)
    self.nStep = target:size(2) 
    self.vocab_size = input[1][1]:size(2)-1 -- first [1] is the 1 time step; second [1] is the 1 element;
					  -- size(2) is the vocab_size+1
    local inputs = utils.reformat(input, self.batch_size, self.nStep)
    self.reward = torch.DoubleTensor(self.batch_size,1):zero()

    for i = 1, self.batch_size do
	print ("sample", i)
	local reward = 0
	local sample = inputs[i]
	local generated_sentence = ""
	local ground_truth_sentence = ""
	for t = 1, self.nStep do
	    self.CiderScorer:reset() -- reset
	    -- ground_truth_sentence.append(word)
	    local ref_word_idx_t = target[i][t]
	    if ref_word_idx_t == 0 then -- if padding 0
		break -- also skip the generated, because we don't want to account for the 0 paddings
	    end
	    local ref_word_t = self.vocab[tostring(ref_word_idx_t)]
	    ground_truth_sentence = ground_truth_sentence .. " " ..  ref_word_t 

	    -- generated_sentence.append(word)
	    local gen_word_prob_t = sample[t] -- generated word probobility at timestep t
	    local _, gen_word_t = torch.max(gen_word_prob_t, 1)
	    gen_word_t = self.vocab[tostring(gen_word_t[1])] --unpack and get the word by index
	    if not gen_word_t then break end -- END token, break the generated sentence here
	    generated_sentence = generated_sentence .. " " .. gen_word_t --append word, insert space in between
	end
	self.CiderScorer:_add(generated_sentence, ground_truth_sentence)
	print ("generated_sentence:", generated_sentence)
	print ("ground_truth_sentence:", ground_truth_sentence)
	reward = self.CiderScorer:compute_score()
	self.reward[i] = reward
    end

    self.reward:mul(self.scale)
    self.output = -self.reward:sum()
    --self.output = reward -- or self.output = -reward ??
    return self.output
end

function VRCIDErReward:updateGradInput(inputTable, target)
    local input22_t1 = inputTable[1][2][2]
    local baseline_reward = torch.DoubleTensor(self.batch_size,1):zero()
    for k, v in pairs(inputTable) do
	if k ~= 1 then 
	    local input22 = v[2][2]
	    assert (utils.roundToNthDecimal(input22[1][1]) == utils.roundToNthDecimal(input22_t1[1][1])) -- error if baseline rewards does not match in each timestep
	end
    end

    local baseline_reward = inputTable[self.nStep][2][2]
    self.vrReward = self.vrReward or self.reward.new()
    self.vrReward:resizeAs(self.reward):copy(self.reward)
    self.vrReward:add(-1, baseline_reward)

    -- broadcast reward to modules
    self.module:reinforce(self.vrReward)

    self.gradInput[1] = torch.DoubleTensor(self.batch_size, self.vocab_size+1):zero()
    self.gradInput[2] = torch.DoubleTensor(self.batch_size, 1):zero()
    self.gradInput[2] = self.criterion:backward(baseline_reward, self.reward)

    return self.gradInput
end

