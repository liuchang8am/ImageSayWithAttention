------------------------------------------------------------------------
--[[ PerplexityCaptioner ]]--
-- Feedback
-- Computes perplexity for language models
-- For now, only works with SoftmaxTree
------------------------------------------------------------------------
local PerplexityCaptioner, parent = torch.class("dp.PerplexityCaptioner", "dp.Feedback")
PerplexityCaptioner.isPerplexityCaptioner = true

function PerplexityCaptioner:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, name = xlua.unpack(
      {config},
      'PerplexityCaptioner',
      'Computes perplexity for language models',
      {arg='name', type='string', default='perplexity',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   parent.__init(self, config)
   self._nll = 0
end

function PerplexityCaptioner:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

-- exponential of the mean NLL
function PerplexityCaptioner:perplexity()
   -- divide by number of elements in sequence
   return torch.exp(self._nll / self._n_sample)
end

function PerplexityCaptioner:doneEpoch(report)
   if self._n_sample > 0 and self._verbose then
      print(self._id:toString().." perplexity = "..self:perplexity())
   end
end

function PerplexityCaptioner:add(batch, output, report)
   assert(torch.isTypeOf(batch, 'dp.Batch'), "First argument should be dp.Batch")
   -- table outputs are expected of recurrent neural networks   
   if torch.type(output) == 'table' then
      -- targets aren't a table
      local targets = batch:targets():forward('bt')
      self._n_sample = self._n_sample + targets:nElement()
      local sum = 0
      for i=1,#output do
         local target = targets:select(2,i)
         local act = output[i][1]
         sum = sum + act[target[1]]
      end
      self._nll = self._nll - sum
   else
      self._n_sample = self._n_sample + batch:nSample()
      local act = output
      if not (torch.isTypeOf(act, 'torch.FloatTensor') or torch.isTypeOf(act, 'torch.DoubleTensor')) then
         self._act = self._act or torch.FloatTensor()
         self._act:resize(act:size()):copy(act)
         act = self._act
      end
      if output:dim() == 2 then
         -- assume output originates from LogSoftMax
         local targets = batch:targets():forward('b')
         print ("targets:", targets)
         io.read(1)
         local sum = 0
         for i=1,targets:size(1) do
            sum = sum + act[i][targets[i]]
         end
         self._nll = self._nll - sum
      else
         -- assume output originates from SoftMaxTree (which is loglikelihood)
         -- accumulate the sum of negative log likelihoods
         self._nll = self._nll - act:view(-1):sum()
      end
   end
end

function PerplexityCaptioner:_reset()
   self._nll = 0
end

function PerplexityCaptioner:report()
   return {
      [self:name()] = {
         ppl = self._n_sample > 0 and self:perplexity() or 0
      },
      n_sample = self._n_sample
   }
end
