require("mobdebug").start()
require 'dp'
require 'rnn'
require 'optim'

------------------------
--- Some util functions
------------------------
function print_info(X)
  -- print the basic information of X
  print (torch.type(X))
  if (torch.type(X)) == 'table' then
    print (" --> table len: ", table.getn(X))
    print (" --> table[i]:")
    print ("          --> type:", torch.type(X[1]))
    print ("          --> len:", table.getn(X[1]))
  else
    print (X)
  end
end

function print_object_member(X)
  for k,v in pairs(X) do
    print(k)
  end
end

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', '/Users/lc/save/1.dat', 'path to a previously saved model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--evalTest', false, 'model was saved with cuda')
cmd:option('--stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('--dataset', 'Flickr8k', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('--overwrite', false, 'overwrite checkpoint')
cmd:text()

local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)
model = xp:model().module 
ra = model:findModules('nn.RecurrentAttention')[1]
--actions_1 = ra.actions
print (model)

model:training()

ds = dp['Flickr8k']()
inputs = ds:get('test','input')
input = inputs:narrow(1,1,3) -- inputs:narrow(1,1,N) --> forward N images
output = model:forward(input)
--actions_2 = ra.actions -- actions_2 is the same with actions_1
actions = ra.actions
output = model.output
--print_info(output)

--print_object_member(ds)

for t = 1, table.getn(output) do -- for each time step, total timesteps = table len
  _, idx = torch.max(output[t][1],2) -- idx is: batch x indexes, return the max_index of each sample
  print (idx)
  Z = 1
end

Z = 1
print ('test')

