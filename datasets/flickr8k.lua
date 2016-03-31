------------------------------------------------------------------------
--[[ Flickr8k ]] --
-- A image caption dataset.
-- Dataformat: 1 image / 5 sentences
------------------------------------------------------------------------
require 'hdf5'

version = 2.0 -- without dp

local Flickr8k = torch.class("Flickr8k")

function Flickr8k:__init(config)
    self._name = 'flickr8k'
    self._image_size = { 3, 256, 256 }
    self._image_axes = 'bchw'
    self._feature_size = 3 * 256 * 256

    config = config or {}
    assert(torch.type(config) == 'table' and not config[1],
        "Constructor requires key-value arguments")
    local args, load_all, input_preprocess, target_preprocess
    args, self._valid_ratio, self._train_file, self._test_file,
    self._data_path, self._scale, self._binarize, self._shuffle,
    self._download_url, load_all
    = xlua.unpack({ config },
        'Flickr8k',
        'Caption Dataset 5 sentences/image',
        {arg = 'valid_ratio', type = 'number',default = 1 / 6},
        {arg = 'train_file',type = 'string',default = 'train.th7'},
        {arg = 'test_file',type = 'string',default = 'test.th7'},
        {arg = 'data_path',type = 'string',default = dp.DATA_DIR .. '/flickr8k/'},
        {arg = 'scale',type = 'table'},
        {arg = 'binarize',type = 'boolean',default = false},
        {arg = 'shuffle',type = 'boolean',default = false},
        {arg = 'download_url',type = 'string',default = 'https://stife076.files.wordpress.com/2015/02/mnist4.zip'},
        {arg = 'load_all',type = 'boolean',default = true},)

    if (self._scale == nil) then
        self._scale = { 0, 1 }
    end
    if load_all then
        self:setup()
        self:createClasses()
        self:loadTrain()
        self:loadValid()
        self:loadTest()
    end
end

--- Load the files
function Flickr8k:setup()
    --1. read data file
    --print("Loading flickr8k data.h5 file ...")
    --local h5_filepath = self._data_path .. '/' .. 'data_t=5.h5'
    self.debug = true
    local h5_filepath
    if self.debug then
	h5_filepath = self._data_path .. '/' .. 'debug100' .. '/' .. 'data_t=5.h5'
	print ("Loading debug_100")
    else
	print("Loading flickr8k data_t=5.h5 file ...")
        h5_filepath = self._data_path .. '/' .. 'data_t=5.h5'
    end

    if self:exists(h5_filepath) then print("Done.") else return end
    local h5_file = hdf5.open(h5_filepath)
    local images = h5_file:read('/images'):all()
    local sentences = h5_file:read('/labels'):all()
    h5_file:close()

    --2. read miscs
    --print("Loading flickr8k data.json file ...")
    --local json_filepath = self._data_path .. '/' .. 'data.json'
    print("Loading flickr8k data_t=5.json file ...")
    local json_filepath = self._data_path .. '/' .. 'data_t=5.json'

    if self.debug then
	json_filepath = self._data_path .. '/' .. 'debug100' .. '/' .. 'data_t=5.json' --overwirte if debug
    end

    if self:exists(json_filepath) then print("Done.") else return end local json_file = io.open(json_filepath)
    local text = json_file:read()
    local cjson = require 'cjson'
    local json_file_contents = cjson.decode(text)
    local ix_to_word = json_file_contents.ix_to_word
    local images_info = json_file_contents.images
    json_file:close()

    self._flickr8k = { images, sentences, images_info, ix_to_word }

    self.vocab = self._flickr8k[4]

    -- add the end token
    self.vocab_size = self:len(self.vocab)+1 -- +1 for the '.' end token
    self.vocab[tostring(self.vocab_size)] = '.' -- use '.' as the end token

    -- add the special token to replace the null tokens
    self.vocab_size = self.vocab_size + 1
    self.vocab[tostring(self.vocab_size)] = '#' -- use '#' as the replacement for null token

    self._traindata = self:_setup_train()
    self._valdata = self:_setup_val()
    self._testdata = self:_setup_test()
end

function Flickr8k:_setup_train()
    return self:_setup_wrapper('train')
end

function Flickr8k:_setup_val()
    return self:_setup_wrapper('val')
end

function Flickr8k:_setup_test()
    return self:_setup_wrapper('val')
end

function Flickr8k:_setup_wrapper(which_set)
    assert(which_set == 'train' or which_set == 'val' or which_set == 'test',
        'Error. Not supported split.')
    local flickr8k = self._flickr8k
    local images = flickr8k[1] -- flickr8k[1][i] -> image i
    local images_info = flickr8k[3] -- flickr8k[3][id]{['file_path'],['split']} 
    local sentences = flickr8k[2] -- flickr8k[2][id][i] -> word i

    local split = {} -- {{inputs}, {targets}}
    local inputs = {} -- 100x3x256x256 images
    local targets = {} -- 100x(sentences) sentences
    local idx = 1

    for i = 1, images:size()[1] do
        if images_info[i]['split'] == which_set then
            -- setting up images
            inputs[idx] = images[i]:float()
            -- setting up sentences
            targets[idx] = sentences[i]
            idx = idx + 1
        end
    end
    local splitSize
    if which_set == 'train' then
        self._trainSize = idx - 1
        splitSize = self._trainSize
    elseif which_set == 'val' then
        self._valSize = idx - 1
        splitSize = self._valSize
    else
        self._testSize = idx - 1
        splitSize = self._testSize
    end
    local str = string.format('Has %d %s images', splitSize, which_set)
    print(str)

    -- convert inputs from table {image1, image2, ... } to tensor splitsizex3x256x256
    local img_channel = inputs[1]:size()[1]
    local img_height = inputs[1]:size()[2]
    local img_width = inputs[1]:size()[3]

    local images_tensor = torch.FloatTensor(splitSize, img_channel, img_height, img_width)
    for i = 1, splitSize do
        images_tensor[i] = inputs[i]
    end

    -- converts sentences to tensor splitSizex16 
    local sentences_tensor = torch.FloatTensor(splitSize, sentences[2]:size()[1])
    for i = 1, splitSize do
        sentences_tensor[i] = targets[i]
    end

    split = { images_tensor, sentences_tensor } -- if return torch.FloatTensor type

    return split
end

function Flickr8k:loadTrain()
    local train_data = self._traindata
    self:trainSet(self:createDataSet(train_data[1], train_data[2]))
    return self:trainSet()
end

function Flickr8k:loadValid()
    local val_data = self._valdata
    self:validSet(self:createDataSet(val_data[1], val_data[2]))
    return self:validSet()
end

function Flickr8k:loadTest()
    local test_data = self._testdata
    self:testSet(self:createDataSet(test_data[1], test_data[2]))
    return self:testSet()
end

function Flickr8k:createDataSet(inputs, targets, which_set)
    if self._shuffle then
        local indices = torch.randperm(inputs:size(1)):long()
        inputs = inputs:index(1, indices)
        targets = targets:index(1, indices)
    end
    -- construct inputs and targets dp.Views
    local input_v, target_v = dp.ImageView(), dp.ClassView()
    input_v:forward(self._image_axes, inputs)
    
    -- add the '.' end token for each sample
    local end_token_column = torch.FloatTensor(targets:size()[1], 1):fill(self.vocab_size) --'#'
    targets = torch.cat(targets,end_token_column) --add a column of '#' 

    -- replace all the zeros with '#'
    for sample = 1, targets:size()[1] do -- for each sample
	for word = 1, targets[sample]:size()[1] do --for each word in the sample
	    if targets[sample][word] == 0 then
		targets[sample][word] = self.vocab_size
	    end
	end
    end

    -- replace the first '#' with '.'
    for sample = 1, targets:size()[1] do -- for each sample
	for word = 1, targets[sample]:size()[1] do --for each word in the sample
	    if targets[sample][word] == self.vocab_size then
		targets[sample][word] = self.vocab_size - 1 -- replace with '.'
		break -- keep the remaining '#' s 
	    end
	end
    end

    -- construct dataset
    local ds = dp.DataSet { inputs = input_v, targets = target_v, which_set = which_set }
    return ds
end

--function Flickr8k:createClasses()
--    self._classes = {}
--    local len = self:len(self._flickr8k[4])
--    self._classes[1] = 0
--    for i = 1, len do
--        self._classes[i+1] = i
--    end
--    return self._classes
--end

function Flickr8k:createClasses()
    self._classes = {}
    local len = self:len(self._flickr8k[4])
    for i = 1, len do
        self._classes[i] = i
    end
    return self._classes
end

---------------------
--- Utils
---------------------
function Flickr8k:isFile(name)
    local f = io.open(name, "r")
    if f ~= nil then
        io.close(f)
        return true
    else
        return false
    end
end

function Flickr8k:exists(filepath)
    if self:isFile(filepath) then
        return true
    else
        str = string.format("Error. File %s does not exist. Please check.", filepath)
        print(str)
        return false
    end
end

function Flickr8k:len(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end
