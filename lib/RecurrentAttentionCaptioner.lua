local RecurrentAttentionCaptioner, parent = torch.class("nn.RecurrentAttentionCaptioner", "nn.Container")

function RecurrentAttentionCaptioner:__init(rnn, action, nStep, hiddenSize)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   
   -- samples an x,y actions for each example
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action}
   
   self.output = {} -- rnn output
   self.actions = {} -- action output
   
   self.forwardActions = false
   
   self.gradHidden = {}
end

function RecurrentAttentionCaptioner:updateOutput(inputs)
   self.rnn:forget()
   self.action:forget()

   local input = inputs[1]:double() -- input is BCHW raw images
   local words = inputs[2]:double() -- words

   local nDim = input:dim()

   words[torch.eq(words,0)] = 1 -- replace 0 with 1, to prevent nn.LookupTable 0 crash
                                -- this operation will not influence the optimization
				-- since we will set the correspoding loss to zero during criterion
   
   --print ("RecurrentAttentionCaptioner self.nStep is", self.nStep)

   for step=1,self.nStep do --self.nStep is opt.rho value
      
      --print ("RecurrentAttentionCaptioner step = ", step) --io.read(1)
      
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
	 --print (self._initInput)
	 --print ("self._initInput")
	 --io.read(1)
	 --print (self.action)
	 --io.read(1)
         self.actions[1] = self.action:updateOutput(self._initInput)
	 --print ("self.actions[1]")
	 --print (self.actions[1])
	 --io.read(1)
      else
         -- sample actions from previous hidden activation (rnn output)
         self.actions[step] = self.action:updateOutput(self.output[step-1])
      end
      
      -- rnn handles the recurrence internally
      word = words[{{}, {step}}] -- select word [i]
      local output = self.rnn:updateOutput{input, self.actions[step], word}
      --self.output[step] = self.forwardActions and {output, self.actions[step]} or output
      self.output[step] = output
   end

  -- print (self.output)
  -- print ("up is self.output in RecurrentAttentionCaptioner")
  -- io.read(1)

   return self.output
end

function RecurrentAttentionCaptioner:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   print (gradOutput)
   print ("gradOutput in RecurrentAttentionCaptioner")
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      print (gradOutput_)
      print ("gradOutput_")
      print (gradAction_)
      print ("gradAction_")
      io.read(1)

      if self.forwardActions then
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action.output.new()
         if not self._gradAction:isSameSizeAs(self.action.output) then
            self._gradAction:resizeAs(self.action.output):zero()
         end
         gradAction_ = self._gradAction
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:updateGradInput(self._initInput, gradAction_)
      else
         local gradAction = self.action:updateGradInput(self.output[step-1], gradAction_)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
      end
      
      -- 2. backward through the rnn layer
      local gradInput = self.rnn:updateGradInput({input, self.actions[step]}, self.gradHidden[step])[1]
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function RecurrentAttentionCaptioner:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
            
      if step == 1 then
         -- backward through initial starting actions
         self.action:accGradParameters(self._initInput, gradAction_, scale)
      else
         self.action:accGradParameters(self.output[step-1], gradAction_, scale)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accGradParameters({input, self.actions[step]}, self.gradHidden[step], scale)
   end
end

function RecurrentAttentionCaptioner:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:accUpdateGradParameters(self._initInput, gradAction_, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         self.action:accUpdateGradParameters(self.output[step-1], gradAction_, lr)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accUpdateGradParameters({input, self.actions[step]}, self.gradHidden[step], lr)
   end
end

function RecurrentAttentionCaptioner:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RecurrentAttentionCaptioner:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
