-- Load images and captions from h5py file --
-- By: LC --
require 'hdf5'
local utils = require 'misc.utils'

function loadData(h5py_file)
  print ("function loaddata")
  print (#h5py_file)
  return utils.read_json(h5py_file)
end
