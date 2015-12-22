require "misc.loadData"
require "misc.utils"

-----------------------------------------------------------------------------
-- Load Dataset, pre-prepared into h5py file
-----------------------------------------------------------------------------
h5_datapath = "./data/flickr8k/data.h5"

local utils = require "misc.utils"
-- check if file exists
if utils.file_exists(h5_datapath) then print ("true") else print ("faluse") end
-- load h5_data
-- data formats:

h5_data = loadData(h5_datapath)