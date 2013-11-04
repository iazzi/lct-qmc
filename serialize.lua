#!/usr/bin/lua

local function serialize (f, o)
	local flush = false
	if f==nil then return end
	if type(f)=='string' then f = assert(io.open(f, 'w')) f:write("return ") flush = true end
	local t = type(o)
	if t=='number' then
		f:write(string.format('%a', o))
	elseif t=='string' then
		f:write(string.format('%q', o))
	elseif t=='boolean' then
		f:write(o and 'true' or 'false')
	elseif t=='table' then
		f:write('{\n')
		for k, v in pairs(o) do
			if type(k)=='string' and k:match('^[_%a][_%w]*$') then
				f:write(' ', k, ' = ')
			else
				f:write(' [') serialize(f, k) f:write('] = ')
			end
			serialize(f, v)
			f:write(',\n')
		end
		f:write('}')
	else
		io.stderr:write("Unknown type `"..t.."`")
	end
	if flush then f:flush() end
end

return serialize

