------------------------------------
----------- Author @ LC ------------
------------------------------------

--require("mobdebug").start()
require 'hdf5' -- read .h5 file format
require 'cjson' -- read .json file format
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
    local h5_file = hdf5.open(datapath)
    local images = h5_file:read('/images'):all() -- read all 100 images, access via images[i] for image i
    local sentences = h5_file:read('/labels'):all() -- read all 100 sentences, access via sentences[i] for image i
    h5_file:close()

    -- 2. load the word-number mapping json file
    -- pre-prepared with neuraltalk2/prepo.py

    -- data.json file format:
    --	-> 'images': containes info of the 100 images
    --	     -> 'file_path': string, the filename, eg. '1597557856_30640e0b43.jpg'
    --	     -> 'split': string, 'train', 'validation', or 'test' split
    --	-> 'ix_to_word': the mapping from number to word, eg. 101 is 'cigarettes' 
    -- #TODO: add assertion to make sure file exists
    print ('Loading flickr8k data.json file ...')
    local json_file = io.open('data.json', 'r'):read()

    -- let's try to print the caption sentence for image[1]


    return h5_file
end


data = prepare_flickr8k('data.h5')
print (A)
