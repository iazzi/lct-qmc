#!/usr/bin/lua

local conf, dir = ...
local t = dofile(conf)
dir = '/cluster/scratch_xl/public/miazzi/'..dir..'/'
local prog = io.popen('pwd'):read('*l')..'/stablefast'

assert(os.execute('mkdir '..dir))
os.execute('cp '..conf..' '..dir..'config.lua')
os.execute('cp serialize.lua '..dir)
for i, c in ipairs(t) do
	--sanitize
	if c.T and not c.beta then
		c.beta = 1.0/c.T
	end
	--write out
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
	f:write('    savefile = "'..dir..tostring(i)..'.save",\n')
	f:write('    outfile = "'..dir..tostring(i)..'.out",\n')
	f:write('  },\n')
	f:write('}\n')
	f:close()
end


