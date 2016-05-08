-- Given one image with (generated / ground truth pair)
-- return the BLEU4 score

-- Modified from https://github.com/tylin/coco-caption/tree/master/pycocoevalcap/bleu

local BleuScorer = torch.class('BleuScorer')
function BleuScorer:__init(arg)
    self.n = 4
    self.cref = {}
    self.ctest = {}
    self.len_cref = 0
    self.len_ctest = 0
    self.special_reflen = nil
    self.score = 0
    self.single_sample = arg.single_sample
    self.reward_signal = arg.reward_signal
end

function BleuScorer:reset()
    self.cref = {}
    self.ctest = {}
    self.len_ctest = 0
    self.len_cref = 0
    self.score = 0
    self.special_reflen = nil
end

function BleuScorer:precook(s) --ngram
    local words = {}
    local counts = {}
    for word in string.gmatch(s, "%S+") do
	table.insert(words, word)
    end
    --print ("words", words)
    for k = 1, self.n do
	for i = 1, table.getn(words)-k+1 do
	    local str = words[i] 
	    for j = i+1, i+k-1 do
		str =  str .. " " .. words[j]
	    end
	    if not counts[str] then 
		counts[str] = 1
	    else -- if has the keys
	        counts[str] = counts[str] + 1
	    end
	end
    end
    return table.getn(words),  counts
end

function BleuScorer:cook_ref(ref)
    local reflen = {}
    local maxcounts = {}
    local rl, counts = self:precook(ref)
    table.insert(reflen, rl)
    for k,v in pairs(counts) do
	if not maxcounts[k] then 
	    maxcounts[k] = v
	else
	    maxcounts[k] = math.max(maxcounts[k], v)
	end
    end
    return reflen, maxcounts 
end

function BleuScorer:cook_test(test, ref)
    local refmaxcounts = ref[1][2]
    local testlen, counts = self:precook(test)
    local result = {}
    result["reflen"] = ref[1][1]
    result["testlen"] = testlen
    result["guess"] = torch.FloatTensor(self.n):zero()
    result["correct"] = torch.FloatTensor(self.n):zero()
    for k = 1,self.n do 
	result["guess"][k] = math.max(0, testlen-k+1)
    end
    for k,v in pairs(counts) do
	local temp = 0
	if refmaxcounts[k] then temp = refmaxcounts[k] end
	result['correct'][self:count_word(k)] 
	    = result['correct'][self:count_word(k)] + math.min(temp, v)
    end
    return result
end

function BleuScorer:cook_append(test, ref)
    if ref then 
	table.insert(self.cref, {self:cook_ref(ref)})
	if test then
	    table.insert(self.ctest, self:cook_test(test, self.cref))
	end
    end
end

function BleuScorer:_add(hypo, ref)
    self:cook_append(hypo, ref)
end

function BleuScorer:_single_reflen(reflens, option, testlen)
    local reflen = 0
    if option == "shortest" then
	reflen = min(reflens)
    elseif option == "average" then
	reflen = torch.mean(reflens)
    elseif option == "closest" then
	reflen = reflens[1] -- #NOTE: only works with one ref now
    end
    return reflen
end

function BleuScorer:compute_bleu(option)
    local small = 1e-9
    local tiny = 1e-15
    local bleu_list = torch.FloatTensor(self.n):zero()

    local totalcomps = {}
    totalcomps['testlen'] = 0
    totalcomps['reflen'] = 0
    totalcomps['guess'] = torch.FloatTensor(self.n):zero()
    totalcomps['correct'] = torch.FloatTensor(self.n):zero()

    for k,comps in pairs(self.ctest) do
	local testlen = comps['testlen']
	self.len_ctest = self.len_ctest + testlen
	local reflen = 0
	if not self.special_reflen then 
	    reflen = self:_single_reflen(comps['reflen'], option, testlen)
	else
	    reflen = self.special_reflen
	end
	self.len_cref = self.len_cref + reflen
	for _, key in pairs({"guess", "correct"}) do
	    for k = 1, self.n do
		totalcomps[key][k] = totalcomps[key][k] + comps[key][k]
	    end
	end

	local bleu = 1
	for k = 1, self.n do
	    bleu = bleu * (comps['correct'][k] + tiny) / (comps['guess'][k] + small)
	    bleu_list[k] = math.pow(bleu, (1/(k)))
	end
	local ratio = (testlen + tiny) / (reflen + small)
	if ratio < 1 then
	    for k = 1, self.n do
		bleu_list[k] = bleu_list[k] * math.exp(1-1/ratio)
	    end
	end
    end
    if self.single_sample then 
	local score = bleu_list
	return score
    else
	--#TODO: if has more than one refs?
	totalcomps['reflen'] = self._reflen
    	totalcomps['testlen'] = self._testlen

    	local bleus = {}
    	local bleu = 1
    	--for k = 1, self.n do
    end
    return self.score 
end


------------------Core Function-----------------
--- Before calling compute_score
--- call "BleuScorer:_add(hypo, ref)" first
function BleuScorer:compute_score()
    -- compute blue score
    local score = self:compute_bleu("closest")
    if self.reward_signal == 5 then -- BLEU_avg
	return torch.mean(score)
    else
	return score[self.reward_signal]
    end
end
-----------------Core Function Ends-------------


---- Below is utility functions ----
function BleuScorer:test()
    local str = "several people seated in chairs in a waiting room with a snapple vending machine in the far corner of the room"
    --self:precook("tall flower arrangement in a pitcher by a window")
    self:precook(str)
end

function BleuScorer:defaultdict(default_value_factory)
    local t = {}
    local metatable = {}
    metatable.__index = function(t, key)
	if not rawget(t, key) then
	    rawset(t, key, default_value_factory(key))
	end
	return rawget(t, key)
    end
    return setmetatable(t, metatable)
   -- local d = self:defaultdict(function () return {} end)
   -- table.insert(d["people", "dog"], {"bob", "mike"})
   -- print (d) io.read(1)
end

function BleuScorer:getLen(t)
-- given a table t, return the number of elements of it
-- use this func when table.getn fails
-- e.g., t = {"dog":1, 2:"cat"}, return 2
    local len = 0
    for k,v in pairs(t) do
	len = len+1
    end
    return len 
end

function BleuScorer:count_word(str)
-- given a string, return the number of words in it
-- e.g, "a dog running", return 3
    local count = 0
    for word in string.gmatch(str,"%S+") do
	count = count + 1
    end
    return count
end

function BleuScorer:mean(T)
-- compute the mean value for table T, elements in T should be numbers
-- e.g., T = {1, 2, 3, 4}, return 2.5
    local count = 0
    local sum = 0
    for k, v in pairs(T) do
	sum = sum + v
	count = count + 1
    end
    sum = sum / count
    return sum
end
