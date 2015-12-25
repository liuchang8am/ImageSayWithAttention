--require('mobdebug').start()
require 'misc/utils'
require 'datasets.Flickr8k'
require 'lfs'
require 'torch'
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-dataset', 'flickr8k', 'which dataset to train. flickr8k, flickr30k or mscoco')

local opt = cmd:parse(arg)

dataset_path = lfs.currentdir()..'/data/'..opt.dataset

print (dataset_path) 

ds = Flickr8k(dataset_path):setup()

print (ds)
