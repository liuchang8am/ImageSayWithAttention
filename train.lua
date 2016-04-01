--require('mobdebug').start()

--packages from torch
require 'rnn'
require 'image'
require 'dpnn'

--user-defined packages
--require 'lib/VRCIDErReward' -- variance reduced CIDEr reward
require 'misc.DataLoader' -- load dataset 'flickr8k', Ffickr30k or coco 
require 'lib/RecurrentAttentionCaptioner'

local debug = true

-------------------------------------------
--- command line parameters
-------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

--- training options ---

cmd:option('--dataset', 'Flickr8k', 'training on which dataset? Flickr8k, Flickr30k or MSCOCO')
cmd:option('--learningRate', 0.001, 'learning rate at t=0')
cmd:option('--lr_decay_every_iter', 10000, 'decay learning rate every __ iter, by opt.lr_decay_factor')
cmd:option('--lr_decay_factor', 10, 'decay learning rate by __')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 2, 'number of examples per batch')
cmd:option('--gpuid', -1, 'sets the device (GPU) to use. -1 = CPU')
cmd:option('--max_iters', -1, 'maximum iterations to run, -1 = forever')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--cv', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--eval_every_iter', 2000, 'eval on validation set every __ iter')
cmd:option('--eval_use_image', 100, 'eval using __ images in validation set')

-- reinforce
cmd:option('--rewardScale', 10, "scale of positive reward (negative is 0)")
cmd:option('--unitPixels', 127, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
cmd:option('--locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

-- model info
cmd:option('--glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', 32, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

-- activate function
cmd:option('--transfer', 'ReLU', 'activation function')

-- language model
cmd:option('--rho',17)
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

local ds = DataLoader{h5_file=input_h5_file, json_file=input_json_file}

--- Define the Model ---
-- model is an agent interacting with the environment(image)
-- it tries to maximize its reward (CIDEr value)
-- training using REINFORCE rule, 
-- as well as surpervised sentence information to provide CIDEr reward

-- 1. location sensor
locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2)) -- select {x,y}
locationSensor:add(nn.Linear(2,opt.locatorHiddenSize))
locationSensor:add(nn[opt.transfer]())

-- 2.glimpse sensor
glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float(), true))
glimpseSensor:add(nn.Collapse(3))
--glimpseSensor:add(nn.Debug())
glimpseSensor:add(nn.Linear(ds:imageSize('c')*opt.glimpsePatchSize^2*opt.glimpseDepth, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

--- 3.glimpse
glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
glimpse:add(nn.Linear(opt.locatorHiddenSize+opt.glimpseHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())
glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

-- 4.words embedding
wordsEmbedding = nn.Sequential()
local lookup = nn.LookupTable(ds:getVocabSize()+1, opt.hiddenSize)
lookup.maxnormout = -1
wordsEmbedding:add(lookup)
-- wordsEmbedding:add(nn.SplitTable(1)) ???? why SplitTable?


-- 5. recurrent layer
if opt.FastLSTM then 
    recurrent = nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)
else
    recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
end

-- 6. recurrent neural network
rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)

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
seq:add(nn.Constant(1,1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

-- output will be : {classpred, {classpred, basereward}}
--agent:add(concat2)
agent:add(nn.Sequencer(concat2))


-- if GPU then convert everything to cuda(), if possible
if opt.gpuid > 0 then
    require 'cutorch'
    require 'cunn'

end

if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--- Set the Cirterion ---
--local crit1 = nn.SequencerCriterion(nn.ClassNLLCriterion())
--local crit2 = nn.VRCIDErReward(agent, opt.rewardScale)
--local criterion = nn.ParallelCriterion(true)
--    :add(nn.ModuleCriterion(crit1, nil, nn.Convert()))
--    :add(nn.ModuelCriterion(crit2, nil, nn.Convert()))

local iter = 0
local sumErr = 0
--- Start training! ---
while true do -- run forever until reach max_iters

    -- get a batch, not the actual batch_size that is forwarded is opt.batchSize * seq_per_img
    -- because each image has several (say, 5) sentences
    local batch = ds:getBatch{batch_size=opt.batchSize, split='train', seq_per_img=opt.seq_per_img}
    local inputs = batch.inputs -- inputs[1]: raw images in (B,C,H,W)
				-- inputs[2]: words in number format
    local targets = batch.targets -- targets

    -- forward
    local outputs = agent:forward(inputs)

    -- need to unpack batch, iterate each sample to loss one by one, due to viariant sequence length problem
    -- eventhough we padded zeros to the sequence, we don't want to forwad those zeros
    local loss = criterion:forward(outputs, targets)
    sumErr = sumErr + loss

    -- backward
    local gradOutputs = criterion:backward(putputs, targest)
    agent:zeroGradParameters()
    agent:backward(inputs, gradOutputs)

    -- update parameters
    agent:updageGradParameters(opt.momentum)
    agent:updateParameters(opt.learningRate)
    agent:maxParamNorm(opt.maxOutNorm)


    if iter % 1000 == 0 then
	collectGarbage()
    end

    -- decay the learning rate
    if iter % opt.lr_decay_every_iter == 0 then
	opt.learningRate = opt.learningRate / opt.lr_decay_factor
	opt.learningRate = math.max(opt.learningRate, opt.minLR)
    end
    
    -- cross validation
    if iter % opt.eval_every_iter == 0 then
	-- eval performance on validation set

	-- if get better performanec, save the checkpoint
    end






    if iter == opt.max_iters then -- reach max_iters
	-- save the model
	
    end






end
