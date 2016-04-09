local CiderScorer = torch.class('CiderScorer')

function CiderScorer:__init()
    self.n = 4 -- n gram
    self.sigma = 6.0
    self.crefs = {}
    self.ctest = {}
    self.document_frequency = {}
    self.len_crefs = 0
    self.len_ctest = 0
    self.ref_len = 0.0 -- log len
end

function CiderScorer:precook(s) --ngram
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
    --print ("=============")
    --print (counts)
    --print (self:getLen(counts))
    --io.read(1)
    return counts
end

function CiderScorer:_add(hypo, ref)
    if ref then
	self:append_crefs(ref)
	if hypo then
	    self:append_ctest(hypo)
	end
    end
    self.len_crefs = self:getLen(self.crefs)
    self.len_ctest = self:getLen(self.ctest)
end

function CiderScorer:append_crefs(ref)
    table.insert(self.crefs, self:precook(ref))
end

function CiderScorer:append_ctest(hypo)
    table.insert(self.ctest, self:precook(hypo))
end

function CiderScorer:compute_doc_freq()
    for idx, ref in pairs(self.crefs) do
	for k, v in pairs(ref) do
	    if not self.document_frequency[k] then
		self.document_frequency[k] = 1
	    else
		self.document_frequency[k] = self.document_frequency[k] + 1
	    end
	end
    end
end

function CiderScorer:compute_cider()
    function counts2vec(cnts)
	local vec = {}
	local norm = torch.Tensor(self.n):zero()
	local length = 0
	for i = 1, self.n do
	    table.insert(vec,{})
	end
	for k, v in pairs(cnts) do -- k: string of words, v: frequency
	    local df = 0.0
	    if self.document_frequency[k] then 
		df = math.log(math.max(1.0, self.document_frequency[k]))
	    else
		df = math.log(1.0) -- self.document_frequency[k] is nil
	    end
	    local n = self:count_word(k)
	    --print (self:count_word(k))
	    if not vec[n][k] then
		vec[n][k] = v * (self.ref_len-df)
	    else
		print ("Error")
	    end
	    norm[n] = norm[n] + math.pow(vec[n][k], 2)
	    if n == 2 then
		length  = length + v
	    end
	end

	for i = 1, self.n do
	    norm[i] = math.sqrt(norm[i])
	end
	return vec, norm, length
    end

    function sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref)
	local data = length_hyp - length_ref
	local val = torch.FloatTensor(4):zero()
	for n = 1, self.n do
	    for ngram, count in pairs(vec_hyp) do
		val[0] = val[0] + math.min(count, 
	    end
	end
    end

    self.ref_len = math.log(self.len_crefs) 

    local scores = {}
    for k, v in pairs(self.ctest) do
	local test = v
	local refs = self.crefs[k]
	local vec, norm, length = counts2vec(test)
	local score = torch.FloatTensor(4):zero()
	for ref,_ in pairs(refs) do -- current only one ref; possible 5 ref in the future
	    print (ref) io.read(1)
	    local vec_ref, norm_ref, length_ref = counts2vec(ref)
	    score = score + sim(vec, vec_ref, norm, norm_ref, length, length_ref)
	end
	local score_avg = torch.mean(score)
	

    end
end

function CiderScorer:compute_score()
    -- compute idf
    self:compute_doc_freq()
    -- compute cider score
    local score = self:compute_cider()
    return mean(score)
end

function CiderScorer:test()
    local str = "several people seated in chairs in a waiting room with a snapple vending machine in the far corner of the room"
    --self:precook("tall flower arrangement in a pitcher by a window")
    self:precook(str)
end

function CiderScorer:defaultdict(default_value_factory)
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

function CiderScorer:getLen(t)
    local len = 0
    for k,v in pairs(t) do
	len = len+1
    end
    return len 
end

function CiderScorer:count_word(str)
    local count = 0
    for word in string.gmatch(str,"%S+") do
	count = count + 1
    end
    return count
end
