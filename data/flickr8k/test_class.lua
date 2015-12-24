local class = require "class"

local A = class('A')

function A:printA()
    print ("hello")
end

local a = A()
a.printA()
