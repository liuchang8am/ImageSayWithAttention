------------------------------------
----------- Author @ LC ------------
------------------------------------

--require("mobdebug").start()
require 'hdf5' -- read .h5 file format
require 'dp'
require 'torchx' -- for paths.indexdir
require 'image'
---------------------------------------------------------------------------
--- First, let's try to convert the Flickr8k dataset into dp.dataset format
--- Don't know how to deal with the sentence yet.
--- Should the sentence be dealt as multi-labels? Or ?
---------------------------------------------------------------------------

function prepare_flickr8k(datapath)
    
    -- 1. load the images, and the caption sentences
    -- pre-prepared with neuraltalk2/prepo.py, to create lua-readable data

    -- data.h5 file format:
    --  -> 'images': (100, 3, 256, 256), contains 100 resized (3x256x256) images. Note: image 100 is black??
    --	-> 'label_end_ix': 1-indexed numbers, 1-100, no idea what it is for, for now.
    --	-> 'label_length': the sentence length of each image captioned
    --	-> 'label_start_ix': again, 1-indexed numbers, 1-100, no idea what it is for 
    --	-> 'labels': (100, 16), 100 images; 16 as the longest sentence;
    --	->	    each containes a sequence of numbers, representing the numbered word 
    -- #TODO: add assertion to make sure file exists
    print ('Loading flickr8k data.h5 file ...')
    local h5_file = hdf5.open(datapath..'/'..'data.h5')
    local images = h5_file:read('/images'):all() -- read all 100 images, access via images[i] for image i
    local sentences = h5_file:read('/labels'):all() -- read all 100 sentences, access via sentences[i] for image i
    h5_file:close()

    -- 2. load the word-number mapping json file
    -- pre-prepared with neuraltalk2/prepo.py

    -- data.json file format:
    --	-> 'images': containes info of the 100 images, 100 x {'flie_path', 'split'}
    --	     -> 'file_path': string, the filename, eg. '1597557856_30640e0b43.jpg'
    --	     -> 'split': string, 'train', 'validation', or 'test' split
    --	-> 'ix_to_word': the mapping from number to word, eg. 101 is 'cigarettes' 
    -- #TODO: add assertion to make sure file exists
    print ('Loading flickr8k data.json file ...')
    local json_file = io.open(datapath..'/'..'data.json', 'r')
    local text = json_file:read()
    local cjson = require 'cjson' -- read .json file format
    local json_file_contents = cjson.decode(text)
    local ix_to_word = json_file_contents.ix_to_word
    local images_info = json_file_contents.images
    json_file:close()

    -- put the images, sentences, images_info, and ix_to_word all together into a flickr8k dataset
    local flickr8k = {images, sentences, images_info, ix_to_word}
    -- access examples:
    require 'image'
    image.display(flickr8k[1][1]) -- first [1]: the images field of flickr8k; second [1]: the first image of the images
    for i = 1, (flickr8k[2][1]):size()[1] do -- :size()[1] makes sure it is a number rather than a torch.LongStorage of size 1
	io.write (flickr8k[2][1][i], ' ')
    end
    --print (flickr8k[2][1]) -- the sentence of image [1]
    print (flickr8k[3][1]['file_path']) 
    print (flickr8k[3][1]['split']) 
    print (flickr8k[4]["120"])

    -- let's try to print the caption sentence for image[1]
    for i = 1, (flickr8k[2][1]):size()[1]-1 do
	local word_int = flickr8k[2][1][i] -- word i in sentence of image[1]
	local word_str = string.format("%d", word_int) -- convert int to string, using string.format function
	local word = flickr8k[4][word_str]
	io.write (word, ' ')
    end
    print ('.')
    return h5_file
end


data = prepare_flickr8k('/home/lc/codes/ImageSayWithAttention/data/flickr8k/')
