--packages from torch
require 'rnn'
require 'image'
require 'dpnn'
local utils = require 'misc.utils'
--user-defined packages
require 'lib.DataLoader' -- load dataset 'flickr8k', Ffickr30k or coco 
require 'lib.RecurrentAttentionCaptioner'
require 'lib.BLEUReward' -- BLEU reward
require 'lib.LMClassNLLCriterion' -- NLL language loss
require 'misc.LC'
require 'sys'

local debug = true

-------------------------------------------
--- command line parameters
-------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

--- training options ---
cmd:option('--dataset', 'flickr8k', 'training on which dataset? Flickr8k, Flickr30k or MSCOCO')
cmd:option('--learningRate', 0.0001, 'learning rate at t=0')
cmd:option('--lr_decay_every_iter', 10000, 'decay learning rate every __ iter, by opt.lr_decay_factor')
cmd:option('--lr_decay_factor', 10, 'decay learning rate by __')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batch_size', 10, 'number of examples per batch') -- actual batch size is this batchSize * 5, where 5 is 5 sentences / image; this parameter should be >= 1
cmd:option('--validsize', 10, 'number of batch for validation')
cmd:option('--gpuid', -1, 'sets the device (GPU) to use. -1 = CPU')
cmd:option('--max_iters', -1, 'maximum iterations to run, -1 = forever')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--cv', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--eval_every_iter', 2000, 'eval on validation set every __ iter')
cmd:option('--eval_use_image', 100, 'eval using __ images in validation set')
cmd:option('--show_status_per_iter', 10, 'show training status every ? iters (avoid frequent print)')
cmd:option('--save_path', './', 'path to save the trained model')

-- loss
cmd:option('--lamda', 1, 'lamda that balances the two losses, i.e., NLL and Reward')

-- reinforce
cmd:option('--rewardScale', 10, "scale of positive reward (negative is 0)")
cmd:option('--unitPixels', 127, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
cmd:option('--locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')
cmd:option('--reward_signal', 4, "reward_signal can be : 1 --> BLEU1, 2 --> BLEU2, 3 --> BLEU3 , 4 --> BLEU4, and 5 --> BLEU_avg")

-- model info
cmd:option('--glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', 128, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')
cmd:option('--wordsEmbeddingSize', 256, 'size of word embedding')

-- activate function
cmd:option('--transfer', 'ReLU', 'activation function')

-- language model
cmd:option('--rho',16)
cmd:option('--hiddenSize', 256)
cmd:option('--FastLSTM', true, 'using LSTM instead of simple linear rnn unit')
cmd:option('--seq_per_img', 5, 'sentence per image, default is set to 5')

local opt = cmd:parse(arg)

--- Load the dataset ---
local  dataset = opt.dataset

if debug then
    data_path = os.getenv("HOME") .. '/data/' .. dataset:lower() .. '/debug100/'
else
    data_path = os.getenv("HOME") .. '/data/' .. dataset:lower() .. '/'
end

-- load raw data, created using prepo.py
local input_h5_file = data_path .. 'data.h5' -- h5 raw images file
local input_json_file = data_path .. 'data.json' -- json sentences file

local ds = DataLoader{h5_file=input_h5_file, json_file=input_json_file, gpuid=opt.gpuid}

--- Define the Model ---
-- model is an agent interacting with the environment(image)
-- it tries to maximize its reward (CIDEr value)
-- training using REINFORCE rule, -- as well as surpervised sentence information to provide CIDEr reward

-- 1. location sensor
locationSensor = nn.Sequential()
--locationSensor:add(nn.LC())
locationSensor:add(nn.SelectTable(2)) -- select {x,y}
locationSensor:add(nn.Linear(2,opt.locatorHiddenSize))
locationSensor:add(nn[opt.transfer]())

-- 2.glimpse sensor
glimpseSensor = nn.Sequential()
--glimpseSensor:add(nn.LC())
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float(), true))
glimpseSensor:add(nn.Collapse(3))
glimpseSensor:add(nn.Linear(ds:imageSize('c')*opt.glimpsePatchSize^2*opt.glimpseDepth, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

--- 3.glimpse
glimpse = nn.Sequential()
glimpse:add(nn.SelectTable(1)) -- Select the {image, (x,y)}
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
glimpse:add(nn.Linear(opt.locatorHiddenSize+opt.glimpseHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())
--glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

-- 4.words embedding
wordsEmbedding = nn.Sequential()
local lookup = nn.LookupTable(ds:getVocabSize()+1, opt.wordsEmbeddingSize)
lookup.maxnormout = -1
wordsEmbedding:add(nn.SelectTable(2)) -- Select the words
wordsEmbedding:add(lookup)
wordsEmbedding:add(nn.SplitTable(2)) 
wordsEmbedding:add(nn.SelectTable(1))
wordsEmbedding:add(nn[opt.transfer]())

-- 5.multimadalEmbedding
multimodalEmbedding = nn.Sequential()
multimodalEmbedding:add(nn.ConcatTable():add(glimpse):add(wordsEmbedding))
multimodalEmbedding:add(nn.JoinTable(1,1))
multimodalEmbedding:add(nn.Linear(opt.imageHiddenSize+opt.wordsEmbeddingSize, opt.hiddenSize))
multimodalEmbedding:add(nn[opt.transfer]())

-- 6. recurrent layer
if opt.FastLSTM then 
    recurrent = nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)
else
    recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
end

-- 6. recurrent neural network
--rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)
rnn = nn.Recurrent(opt.hiddenSize, multimodalEmbedding, recurrent, nn[opt.transfer](), 99999)
-- nn.Recurrent(start, input, feedback, transfer, rho, merge)
-- 7. action: sample {x,y} using reinforce
locator = nn.Sequential()
locator:add(nn.Linear(opt.hiddenSize,2))
locator:add(nn.HardTanh())
locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
locator:add(nn.MulConstant(opt.unitPixels*2/ds:imageSize("h")))

-- 8. the core: attend to interested places recurrently
attention = nn.RecurrentAttentionCaptioner(rnn, locator, opt.rho, {opt.hiddenSize})

-- 9. the final model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(attention)

-- classifier :
step = nn.Sequential()
step:add(nn.Linear(opt.hiddenSize, ds:getVocabSize()+1))
step:add(nn.LogSoftMax())
agent:add(nn.Sequencer(step))

-- add the baseline reward predictor
seq = nn.Sequential()
seq:add(nn.SelectTable(-1))
seq:add(nn.Constant(1,1))
seq:add(nn.Add(1))

concat = nn.ConcatTable():add(nn.Identity()):add(seq)

-- output will be : {{time*batch*classpred}, batch*basereward}}
agent:add(concat)


if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--- Set the Cirterion ---
local crit1 = nn.LMClassNLLCriterion{vocab=ds.ix_to_word, gpuid=opt.gpuid}
local crit2 = nn.BLEUReward{module=agent, scale=opt.rewardScale, vocab=ds.ix_to_word, reward_signal=opt.reward_signal, gpuid=opt.gpuid}

--print ("Agent:")
--print (agent)

-- if GPU then convert everything to cuda(), if possible
if opt.gpuid < 0 then
    print ("Training in CPU mode ...")
elseif opt.gpuid >= 0 then --#TODO: if GPU enabled, some function may fail in Captioner and LM loss
    require 'cunn'
	require 'cutorch'
	cutorch.setDevice(opt.gpuid)
	agent:cuda()
	crit1:cuda()
	crit2:cuda()
end

local iter = 0
local t = sys.clock()
local time = 0

local results = {}
results.model = agent
results.opt = opt
results.min_valppl = 1e6
--- Start training! ---
while true do -- run forever until reach max_iters
    agent:training()
    sys.tic()

    if iter % opt.show_status_per_iter == 0 then 
	io.write ("===========> Iter:", iter, ' ')
    end
    local sumErr = 0

    -- get a batch, not the actual batch_size that is forwarded is opt.batchSize * seq_per_img
    -- because each image has several (say, 5) sentences
    local batch = ds:getBatch{batch_size=opt.batch_size, split='train', seq_per_img=opt.seq_per_img}
    local inputs = batch.inputs -- inputs[1]: raw images in (B,C,H,W)
				-- inputs[2]: words in number format
    local targets = batch.targets -- targets

    -- forward
    --print ("======> Forward propagation")
    local outputs = agent:forward(inputs) 
					  
    --print ("agent outputs:")
    --print (outputs)
    --io.read(1)

    -- need to unpack batch, iterate each sample to loss one by one, due to viariant sequence length problem
    -- eventhough we padded zeros to the sequence, we don't want to forwad those zeros
    -- unpack done in LMClassNLLCriterion
    --local loss = criterion:forward(outputs, targets)
    local loss1 = crit1:forward(outputs, targets)
    local loss2 = crit2:forward(outputs, targets)
    sumErr = sumErr + loss1 + opt.lamda*loss2

    --io.read(1)
    -- backward
    --print ("======> Back propagation")
    -- backward through multiple loss
    local gradOutput1 = crit1:backward(outputs, targets)
    --print ("gradOutput1:", gradOutput1)
    local gradOutput2 = crit2:backward(outputs, targets)
    --print ("gradOutput2:", gradOutput2)
    local gradOutputs = utils.addGradLosses(gradOutput1, gradOutput2)
    --print ("gradOutputs:", gradOutputs)

    -- backward through the model
    agent:zeroGradParameters()
    agent:backward(inputs, gradOutputs)

    -- update parameters
    agent:updateGradParameters(opt.momentum)
    agent:updateParameters(opt.learningRate)
    agent:maxParamNorm(opt.maxOutNorm)

    t = sys.toc()
    time = time + t
    if iter % opt.show_status_per_iter == 0 then 
	io.write('   <per iter costs ', utils.roundToNthDecimal((time/opt.show_status_per_iter),2), "s> \n") 
	io.write ("    loss=", sumErr, '\n')
	time = 0
    end

    if iter % 1000 == 0 then
	collectgarbage()
    end
    iter = iter + 1

    -- decay the learning rate
    if iter % opt.lr_decay_every_iter == 0 then
	opt.learningRate = opt.learningRate / opt.lr_decay_factor
	opt.learningRate = math.max(opt.learningRate, opt.minLR)
    end
    
    -- cross validation & save model
    if iter % opt.eval_every_iter == 0 then
	-- eval performance on validation set
	agent:evaluate()
	local eval_batch = ds:getBatch{batch_size=opt.valid_size, split='val', seq_per_img=opt.seq_per_img}
	local eval_inputs = eval_batch.inputs
	local eval_targets = eval_batch.targets
	local eval_outputs = agent:forward(eval_inputs)
	local eval_loss1 = crit1:forward(eval_inputs)
	local eval_loss2 = crit2:forward(eval_inputs)
	local eval_loss = eval_loss1 + opt.lamda * eval_loss2
	local ppl = torch.exp(eval_loss / opt.validsize)
	print ("Evaluate on val split, getting perplexity of ", ppl) 

	if ppl < results.min_valppl then -- get better checkpoint, save it
	    results.min_valppl = ppl
	    results.iter = iter
	    torch.save(opt.save_path .. 'results.t7', results)
	else
	    print ("Exploding, early stops on iter", iter)
	    break
	end
    end
end
