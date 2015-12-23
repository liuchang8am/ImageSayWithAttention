require('mobdebug').start()
require "dp"

ds = dp['Flickr8k']()
ds.name = "name is Flickr8k"
print (ds.name)