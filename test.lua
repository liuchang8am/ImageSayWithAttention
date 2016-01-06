--require('mobdebug').start()
require 'misc/utils'
require 'datasets.Flickr8k'
require 'lfs'
require 'torch'
require 'rnn'
require 'dp'
require 'image'

ds = dp['Mnist']()

-------------------------------------------
--- command line parameters
-------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

---  dataset info  ---
cmd:option('-dataset', 'flickr8k', 'which dataset to train. flickr8k, flickr30k or mscoco')

---  model info  ---
-- glimpse
cmd:option('--glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', 8, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')


local opt = cmd:parse(arg)

-------------------------------------------
--- setup your dataset
-------------------------------------------
dataset_path = lfs.currentdir()..'/data/'..opt.dataset
ds = Flickr8k(dataset_path):setup()
print (ds:imageSize('c'))

------------------------------------------
train = ds:get('train', 'inputs', 'bchw')
sample_image = train[1]
-- image.display(sample_image)

------------------------------------------
---------------   Model   ----------------
------------------------------------------
function build_network()
--- 1. setup glimpse sensor
---   converts a raw image into three small patches
---   uncomment the image.display() line to see the result patches
---	-> Inputs: a raw image, say 3x256x256 RGB image from Flickr8k
---	-> Outputs: three patches, concatenate into a vector
end
