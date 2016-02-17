------------------------------------------------------------------------
--[[ PropagatorCaptioner ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a module in order to evaluate its criterion, train, etc.
-- To make your own training algorithm, you can build your own 
-- propagator. If you can reduce it to reusable components, you could 
-- then refactor these into visitors, observers, etc.
------------------------------------------------------------------------
local PropagatorCaptioner = torch.class("dp.PropagatorCaptioner")
PropagatorCaptioner.isPropagatorCaptioner = true

function PropagatorCaptioner:__init(config)   
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, loss, callback, epoch_callback, sampler, observer, 
      feedback, progress, verbose, _cuda, stats = xlua.unpack(
      {config},
      'PropagatorCaptioner', 
      'Propagates Batches sampled from a DataSet using a Sampler '..
      'through a Model in order to evaluate a Loss, provide Feedback '.. 
      'or train the model',
      {arg='loss', type='nn.Criterion',
       help='a neural network Criterion to evaluate or minimize'},
      {arg='callback', type='function',
       help='function(model, report) that does things like'..
       'update model, gather statistics, decay learning rate, etc.'},
      {arg='epoch_callback', type='function', 
       help='function(model, report) that is called between epochs'},
      {arg='sampler', type='dp.Sampler', 
       help='Iterates through a DataSet. [Default=dp.Sampler()]'},
      {arg='observer', type='dp.Observer', 
       help='observer that is informed when an event occurs.'},
      {arg='feedback', type='dp.Feedback',
       help='takes predictions, targets, model and visitor as input '..
       'and provides feedback through report(), setState, or mediator'},
      {arg='progress', type='boolean', default=false, 
       help='display progress bar'},
      {arg='verbose', type='boolean', default=true,
       help='print verbose information'},
      {arg = '_cuda', type='boolean', req=true,help='use gpu or not'},
      {arg='stats', type='boolean', default=false,
       help='display performance statistics (speed, etc). '..
      'Only applies if verbose is true.'}
   )
   self:sampler(sampler or dp.Sampler())
   self:loss(loss)
   self:observer(observer)
   self:feedback(feedback)
   self:callback(callback)
   self:epochCallback(epoch_callback or function() return end)
   self._progress = progress
   self._verbose = verbose
   self._stats = stats
   self._cuda = _cuda
end

function PropagatorCaptioner:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, id, model, mediator, target_module = xlua.unpack(
      {config},
      'PropagatorCaptioner:setup', 
      'Post-initialization setup of the PropagatorCaptioner',
      {arg='id', type='dp.ObjectID', req=true,
       help='uniquely identifies the propagator.'},
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='mediator', type='dp.Mediator', req=true,
       help='used for inter-object communication.'},
      {arg='target_module', type='nn.Module', 
       help='Optional module through which targets can be forwarded'}
   )
   assert(torch.isTypeOf(id, 'dp.ObjectID'))
   self._id = id
   assert(torch.isTypeOf(mediator, 'dp.Mediator'))
   self._mediator = mediator
   self:model(model)
   self._target_module = target_module or nn.Identity()
   self._sampler:setup{mediator=mediator, model=model}
   if self._observer then self._observer:setup{
      mediator=mediator, subject=self
   } end
   if self._feedback then self._feedback:setup{
      mediator=mediator, propagator=self, dataset=dataset
   } end
   if self._visitor then self._visitor:setup{
      mediator=mediator, model=self._model, propagator=self
   } end
end

function PropagatorCaptioner:propagateEpoch(dataset, report)
   self.sumErr = 0
   if self._feedback then
      self._feedback:reset()
   end
   
   -- local vars
   local start_time = sys.clock()
   local batch, i, n, last_n
   local n_batch = 0
   
   if self._stats and self._verbose then
      print('==> epoch # '..(report.epoch + 1)..' for '..self:name()..' :')
   end
   
   if self._model.forget then
      -- for recurrent modules, forget between epochs
      self._model:forget()
   end
   
   self._epoch_callback(self._model, report)
   
   self._n_sample = 0
   local sampler = self._sampler:sampleEpoch(dataset)
   while true do
      -- reuse the batch object
      if batch then
         assert(torch.type(batch) == 'dp.Batch')
      end
      
      batch, i, n = sampler(batch)
      if not batch then 
         -- for aesthetics :
         if self._progress then
            xlua.progress(last_n, last_n)
         end
         break 
      end
      
      self.nSample = i
      self:propagateBatch(batch, report)
      
      if self._progress then
         -- display progress
         xlua.progress(i, n)
      end
      last_n = n
      n_batch = n_batch + 1
   end
   
   -- time taken
   self._epoch_duration = sys.clock() - start_time
   self._batch_duration = self._epoch_duration / math.max(n_batch, 0.000001)
   self._example_speed = last_n / self._epoch_duration
   self._batch_speed = n_batch / self._epoch_duration
   if self._stats and self._verbose then
      print("==> example speed = "..self._example_speed..' examples/s')
   end
end      

function PropagatorCaptioner:propagateBatch(batch)
   error"NotImplementedError"
end

function PropagatorCaptioner:forward(batch)
   local input = batch:inputs():input()
   local target = batch:targets():input()
   temp = target
   for i = 1, 16 do
     print (ds.vocab[tostring(target[1][i])])
    end
   target = self._target_module:forward(target)
   if self._include_target then
      input = {input, target}
   end
   -- useful for calling accUpdateGradParameters in callback function
   self._model.dpnn_input = input
   
   -- forward propagate through model
   self.output = self._model:forward(input)
   --print (target)
   print ('---target---')
   --for i = 1, 16 do
   --   if target[1][i] ~= 1 then
        --io.write (ds._flickr8k[4][tostring(target[1][i])])
        --io.write (' ')
     -- end
   --end
   --io.read(1)
   --io.write ("\n")
   --#TODO: add a module to specifically deal with the data format convertion
   -- change the self.output format, so that it can be dealt with the loss:forward()
   self._temp_output = {}
   self._temp_output[1] = {}
   self._temp_output[2] = {}
   for k,v in pairs(self.output) do
     self._temp_output[1][k] = v[1]
     self._temp_output[2][k] = v[2]
   end
   
   if not self._loss then
      return
   end

   local temp_target
   if self._cuda then 
     temp_target = target:transpose(1,2):cuda()
   else 
     temp_target = target:transpose(1,2)
   end

   self.err = self._loss:forward(self._temp_output, temp_target)
   -- measure loss
   --self.err = self._loss:forward(self.output, target)
end

function PropagatorCaptioner:monitor(batch, report)
   self.sumErr = self.sumErr + (self.err or 0)
   -- monitor error and such
   if self._feedback then
     self._feedback:add(batch, self._temp_output[1], report)
      --self._feedback:add(batch, self.output, report)
   end
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneFeedback", report, batch)
end

function PropagatorCaptioner:doneBatch(report)   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneBatch", report)
end

-- returns a log for the current epoch, in the format of a table
-- or we could create an EpochLog class to help with this.
-- But a table is more flexible. The advantage over pylearn2 is that 
-- the log of an epoch is structured, as opposed to just a list of 
-- channel names and values. Furthermore, values can be anything 
-- serializable.
function PropagatorCaptioner:report()
   local avgErr
   if self._loss and self.sumErr and self.nSample > 0 then
      avgErr = self.sumErr/self.nSample
      if self._verbose then
         print(self:id():toString()..':loss avgErr '..avgErr)
      end
   end
   local report = {
      name = self:id():name(),      
      loss = self._loss and self._loss.report and self._loss:report() or avgErr,
      sampler = self._sampler:report(),
      epoch_duration = self._epoch_duration,
      batch_duration = self._batch_duration,
      example_speed = self._example_speed,
      num_batches = self._num_batches,
      batch_speed = self._batch_speed
   }
   if self._feedback then
      report.feedback = self._feedback:report()
   end
   return report
end

function PropagatorCaptioner:sampler(sampler)
   if sampler then
      self._sampler = sampler
      return
   end
   return self._sampler
end

function PropagatorCaptioner:id(id)
   if id then
      self._id = id
      return
   end
   return self._id
end

function PropagatorCaptioner:name()
   return self._id:name()
end

function PropagatorCaptioner:model(model)
   if model then
      self._model = model
      return
   end
   return self._model
end

function PropagatorCaptioner:observer(observer)
   if observer then
      if torch.type(observer) == 'table' then
         --if list, make composite observer
         observer = dp.CompositeObserver(observer)
      end
      self._observer = observer
      return
   end
   return self._observer
end

function PropagatorCaptioner:callback(callback)
   if callback then
      assert(torch.type(callback) == 'function', "expecting function")
      self._callback = callback
      return
   end
   return self._callback
end

function PropagatorCaptioner:epochCallback(callback)
   if callback then
      assert(torch.type(callback) == 'function', "expecting function")
      self._epoch_callback = callback
      return
   end
   return self._epoch_callback
end

function PropagatorCaptioner:feedback(feedback)
   if feedback then
      if torch.type(feedback) == 'table' then
         --if list, make visitor_chain
         feedback = dp.CompositeFeedback{feedbacks=feedback}
      end
      self._feedback = feedback
      return
   end
   return self._feedback
end

function PropagatorCaptioner:loss(loss)
   if loss then
      assert(torch.isTypeOf(loss, 'nn.Criterion'), "Expecting nn.Criterion instance")
      self._loss = loss
      return
   end
   return self._loss
end

function PropagatorCaptioner:includeTarget(mode)
   -- forward propagates {input, target} instead of just input
   self._include_target = (mode == nil) and true or mode
end

function PropagatorCaptioner:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
   if self._feedback then
      self._feedback:verbose(self._verbose)
   end
   if self._observer then
      self._observer:verbose(self._verbose)
   end
end

function PropagatorCaptioner:silent()
   self:verbose(false)
end

function PropagatorCaptioner:type(new_type)
   if self._loss then
      self._loss:type(new_type)
   end
end

function PropagatorCaptioner:float()
   return self:type('torch.FloatTensor')
end

function PropagatorCaptioner:double()
   return self:type('torch.DoubleTensor')
end

function PropagatorCaptioner:cuda()
   return self:type('torch.CudaTensor')
end

function PropagatorCaptioner:int()
   return self:type('torch.IntTensor')
end

function PropagatorCaptioner:long()
   return self:type('torch.LongTensor')
end
