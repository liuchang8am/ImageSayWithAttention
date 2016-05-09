local LC, _ = torch.class('nn.LC', 'nn.Module')

function LC:updateOutput(input)
   self.output = input
   print (input, "up is input in LC") io.read(1)
   print (input[1], "input [1]") io.read(1)
   print (input[2], "input [2]") io.read(1)
   return self.output
end


function LC:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function LC:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end
