local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

function utils.reformat(inputs, batchSize, nStep, gpuid)
    -- reformat the inputs, i.e., the outputs of the agent model for convinient loop

    local D = inputs[1]:size(2)
	local outputs
    if gpuid >= 0 then 
        outputs = torch.CudaTensor(batchSize, nStep, D)
	else
        outputs = torch.Tensor(batchSize, nStep, D)
	end

    for batch = 1, batchSize do
	    for step = 1, nStep do
	        outputs[{batch, step}] = inputs[step][batch]    
	    end
    end

    return outputs
end


function utils.roundToNthDecimal(num, n)
    local mult = 10^(n or 0)
    return math.floor(num*mult+0.5) / mult
end

function utils.addGradLosses(gradloss1, gradloss2)
    -- loss1 is LM loss, loss2 is reward, add corresponding part
    assert (torch.type(gradloss1) == "table", "LM loss outputs should be table")
    assert (torch.type(gradloss2) == "table", "Reward loss outputs should be table")

    local gradloss = {}
    table.insert(gradloss, gradloss1)
    table.insert(gradloss, gradloss2[2]) -- loss2[1] is always 0
    
    return gradloss
end

function utils.count_word(str)
-- given a string, return the number of words in it
-- e.g, "a dog running", return 3
    local count = 0
    for word in string.gmatch(str,"%S+") do
	count = count + 1
    end
    return count
end

return utils

