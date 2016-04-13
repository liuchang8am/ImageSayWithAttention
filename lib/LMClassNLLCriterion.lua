require 'nn'
local utils = require '../misc/utils'

local LMClassNLLCriterion, parent = torch.class("nn.LMClassNLLCriterion", "nn.Criterion")

function LMClassNLLCriterion:__init()
    parent.__init(self)
    self.criterion = nn.ClassNLLCriterion()
end

function LMClassNLLCriterion:updateOutput(inputs, targets) -- criterion forward

    --print ("LM: updateOutput")
    --print (inputs)
    --print ("up is input")
    --print (targets)
    --print ("up is targets")
    --io.read(1)

    self.nStep = table.getn(inputs)

    self.batchSize = targets:size(1)

    local sum_loss = 0
    local end_token = inputs[1][1]:size(2) --[1] is first timestep, second [1] is first element, size(2) is vocab_size+1

    -- need to reformat the inputs in batch x timestep, rather than timestep x batch
    inputs = utils.reformat(inputs, self.batchSize, self.nStep)

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
        	--print ("target", target)
		loss = self.criterion:forward(input, target)
		sum_loss = sum_loss + loss--accumulate loss
		n = n + 1 -- accumulate if it's a valid loss computation
	    end
	end
    end

    self.output = sum_loss / n
    print ("loss1 NLL:")
    print (self.output)
    return self.output 
end

function LMClassNLLCriterion:updateGradInput(input, targets) -- criterion backward
    -- #TODO
end

--function LMClassNLLCriterion:reformat(inputs)
--    -- reformat the inputs, i.e., the outputs of the agent model for convinient loop
--    local outputs = {}
--    for batch = 1, self.batchSize do
--	local temp  ={}
--	for step = 1, self.nStep do
--	    table.insert(temp,inputs[step][1][batch])
--	end
--	table.insert(outputs, temp)
--    end
--    return outputs
--end
