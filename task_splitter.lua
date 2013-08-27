#!/usr/bin/lua

local conf, dir = ...
local t = dofile(conf)
dir = '/cluster/scratch_xl/public/miazzi/'..dir..'/'

assert(os.execute('mkdir '..dir))
os.execute('cp serialize.lua '..dir)
for i, c in ipairs(t) do
	local f = assert(io.open(dir..tostring(i)..'.in', 'w'))
	f:write("serialize = require 'serialize'\n")
	f:write("return {\n")
	f:write("  {\n")
	for k, v in pairs(c) do
		if type(v)=="string" then
			f:write('    '..k..' = "'..v..'",\n')
		elseif type(v)=="number" then
			f:write('    '..k..' = '..v..',\n')
		else
		end
	end
	f:write('    outfile = "'..dir..tostring(i)..'.out",\n')
	f:write('  },\n')
	f:write('}\n')
	f:close()
end


