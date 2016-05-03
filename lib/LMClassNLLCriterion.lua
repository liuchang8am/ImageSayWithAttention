require 'nn'
local utils = require '../misc/utils'

local LMClassNLLCriterion, parent = torch.class("nn.LMClassNLLCriterion", "nn.Criterion")

function LMClassNLLCriterion:__init(args)
    parent.__init(self)
    self.criterion = nn.ClassNLLCriterion()
    self.vocab = args.vocab
    self.vocab_size = utils.count_keys(self.vocab)
end

function LMClassNLLCriterion:updateOutput(inputTable, targets) -- criterion forward
    
    --print (inputTable)
    --print ("up is inputs")
    --io.read(1) 
    --print (targets)
    --print ("up is targets")
    --io.read(1)
    local inputs = inputTable[1] -- the probs
    self.nStep = table.getn(inputs)

    self.batchSize = targets:size(1)

    local sum_loss = 0
    local end_token = self.vocab_size+1

    -- need to reformat the inputs in batch x timestep, rather than timestep x batch
    inputs = utils.reformat(inputs, self.batchSize, self.nStep)

    --print ("input after reformat")
    --print (inputs)
    --io.read(1)

    -- set the gradInput after the reformat of inputs
    self.gradInput:resizeAs(inputs):zero()

    local n = 0 -- for loss normalization
    for batch = 1, self.batchSize do -- first iterate over batch 
	local first_time = true -- flag used for judging 0 padding
	local sample_inputs = inputs[batch]
	local sample_targets = targets[batch]
	for step = 1, self.nStep do -- then iterate over timestep 
	    local loss
	    local input = sample_inputs[step]
	    local target = sample_targets[step]
	    -- deal with 0 padding; 0 will cause ClassNLLCriterion target > 0 asserton error
	    if target == 0 and first_time then
		target = end_token
		first_time = false
	    end
	    if target ~= 0 then 
		loss = self.criterion:forward(input, target)
		self.gradInput[{batch, step, target}] = -1
		sum_loss = sum_loss - loss--accumulate loss
		n = n + 1 -- accumulate if it's a valid loss computation
	    end
	end
    end
    self.output = sum_loss / n
    self.gradInput:div(n)
    self.gradInput = self:reformat_gradInput(self.gradInput)
    --print (self.output)
    --io.read(1)
    --print (self.gradInput)
    --io.read(1)
    return self.output 
end

function LMClassNLLCriterion:reformat_gradInput(gradInput) -- gradInput is batchSize x nStep x vocab_size
    local output = {}
    for step = 1, self.nStep do
	local temp = gradInput[{{}, step, {}}] -- select 10x156 along the time dimension
	table.insert(output, temp)
    end
    return output
end

function LMClassNLLCriterion:updateGradInput(input, targets) -- criterion backward
    return self.gradInput
end

