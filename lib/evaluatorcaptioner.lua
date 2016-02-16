------------------------------------------------------------------------
--[[ EvaluatorCaptioner ]]--
-- Evaluates (tests) a model using a sampling distribution.
-- For evaluating the generalization of the model, separate the 
-- training data from the test data. The EvaluatorCaptioner can also be used 
-- for early-stoping.
------------------------------------------------------------------------
local EvaluatorCaptioner = torch.class("dp.EvaluatorCaptioner", "dp.PropagatorCaptioner")
EvaluatorCaptioner.isEvaluatorCaptioner = true

function EvaluatorCaptioner:propagateBatch(batch, report) 
   self._model:evaluate()
   self:forward(batch)
   self:monitor(batch, report)
   if self._callback then
      self._callback(self._model, report)
   end
   self:doneBatch(report)
end
