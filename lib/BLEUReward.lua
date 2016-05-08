-- BLUE4 as the reward signal

-- #TODO: if BLEU4 not working because the value is too low,
--	    try BLEU1 or average BLEU


require '../misc/bleu_scorer'
local utils = require '../misc/utils'

local BLEUReward, parent = torch.class("nn.BLEUReward", "nn.Criterion")

function BLEUReward:__init(args)
    parent.__init(self)
    self.module = args.module -- so it can call module:reinforce(reward)
    self.scale = args.scale or 1 -- scale of reward
    self.criterion = nn.MSECriterion() -- baseline criterion --#TODO: check whether should be MSECriterion --> Done, yes
    self.sizeAverage = true
    self.gradInput = {}
    self.reward = {}
    self.vocab = args.vocab -- vocab is ix_to_word
    self.vocab_size = utils.count_keys(self.vocab)
    -- reward_signal can be : 1 --> BLEU1, 2 --> BLEU2, 3 --> BLEU3 , 4 --> BLEU4, and 5 --> BLEU_avg
    self.BleuScorer = BleuScorer{single_sample=true, reward_signal=args.reward_signal}
end

function BLEUReward:int2word(index)
-- given the index of word, return the word in vocab table
    for k, v in pairs(self.vocab) do
	if k == index then
	    return v
	end
    end
end

function BLEUReward:updateOutput(inputTable, target)
    assert(torch.type(inputTable) == 'table')

    --print ("input:", inputTable)
    --print ("target:", target)
    --io.read(1)

    self.batch_size = target:size(1)
    self.nStep = target:size(2) 
    local inputs = inputTable[1] -- fetch the probs; the baseline reward need not to be used during the forward
    local inputs = utils.reformat(inputs, self.batch_size, self.nStep)
    self.reward = torch.DoubleTensor(self.batch_size):zero()

    for i = 1, self.batch_size do
	self.BleuScorer:reset() -- reset --#TODO: should I reset??? --> Done, yes
	--print ("sample", i)
	local reward = 0
	local sample = inputs[i]
	local generated_sentence = ""
	local ground_truth_sentence = ""

	for t = 1, self.nStep do  -- ground_truth_sentence.append(word)
	    --local ref_word_idx_t = target[i][t]
	    ref_word_idx_t = target[i][t]
	    if ref_word_idx_t == 0 then -- if padding 0
		break
	    end
	    local ref_word_t = self.vocab[tostring(ref_word_idx_t)]
	    ground_truth_sentence = ground_truth_sentence .. " " ..  ref_word_t 
	end

	for t = 1, self.nStep do -- generated_sentence.append(word)
	    local gen_word_prob_t = sample[t] -- generated word probobility at timestep t
	    local _, gen_word_idx_t = torch.max(gen_word_prob_t, 1)
	    if gen_word_t == self.vocab_size+1 then print ("Test here") io.read(1)  break end
	    gen_word_t = self.vocab[tostring(gen_word_idx_t[1])] --unpack and get the word by index
	    if gen_word_t then -- only concatenarte if gen_word_t is not vocab_size + 1
		generated_sentence = generated_sentence .. " " .. gen_word_t --append word, insert space in between
	    end
	end

	--generated_sentence = ground_truth_sentence
	self.BleuScorer:_add(generated_sentence, ground_truth_sentence)
	print ("generated_sentence:", generated_sentence, utils.count_word((generated_sentence)))
	print ("ground_truth_sentence:", ground_truth_sentence, utils.count_word(ground_truth_sentence))
	reward = self.BleuScorer:compute_score()
	--print ("reward:", reward)
	--io.read(1)
	if reward >= 0.4 then -- thresh
	    reward = 1
	else
	    reward = 0
	end
	self.reward[i] = reward
    end

    self.reward:mul(self.scale)
    --#TODO: should I maximize the BLEU reward or minimize the -BLEU, according to NLL loss?
    self.output = self.reward:sum() -- or self.output = -reward ??
    --self.output = -self.reward:sum()
    self.output = self.output / self.batch_size
    return self.output
end

function BLEUReward:updateGradInput(inputTable, target)
    assert(torch.type(inputTable) == 'table', "error data type for BLEUReward backward")
    local baseline_reward = inputTable[2] -- fetch the baseline reward
    self.vrReward = self.vrReward or self.reward.new()
    self.vrReward:resizeAs(self.reward):copy(self.reward)
    --print ("____________> baseline reward:")
    --print (baseline_reward)
    self.vrReward:add(-1, baseline_reward)

    -- broadcast reward to modules
    self.module:reinforce(self.vrReward)

    -- format the gradInput
    self.gradInput[1] = {}
    for i = 1, self.nStep do -- 0 grads for the NLL
	table.insert(self.gradInput[1], torch.DoubleTensor(self.batch_size, self.vocab_size+1):zero())
    end

    self.gradInput[2] = torch.DoubleTensor(self.batch_size, 1):zero()
    self.gradInput[2] = self.criterion:backward(baseline_reward, self.reward)

    return self.gradInput
end

