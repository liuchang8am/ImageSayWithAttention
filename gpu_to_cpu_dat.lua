require 'dp'
require 'cunn' 
require 'rnn'
cmd = torch.CmdLine()
cmd:text()
cmd:option('--checkpoint', '/home/lc/save/1.dat', 'path to a previously saved model')
cmd:option('--outfilepath', '/home/lc/save/1_cpu.dat', 'path to the dest save path')
cmd:option('--dataset', 'Flickr8k', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:text()

local opt = cmd:parse(arg or {})

assert(paths.filep(opt.checkpoint), opt.checkpoint..' does not exist')

xp = torch.load(opt.checkpoint)
model = xp:model().module
model_cpu = model:float()
torch.save(opt.outfilepath, model_cpu)
