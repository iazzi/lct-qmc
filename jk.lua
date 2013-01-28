#!/usr/bin/lua

local function transpose (t)
	local ret = {}
	for i, v in ipairs(t) do
		for j, w in ipairs(v) do
			local x = ret[j] or {}
			x[i] = w
			ret[j] = x
		end
	end
	return ret
end

local function average (t)
	local sum = 0
	local n = 0
	for k, v in pairs(t) do
		sum, n = sum + v, n + 1
	end
	return sum / n
end

for _, fn in ipairs{...} do
	local t = {}
	local f = assert(io.open(fn))
	for l in f:lines() do
		if l:match('^%s*$') or l:match('%#.*') then
		else
			local values = { l:match(('(%S+)%s*'):rep(12)) }
			for i, v in ipairs(values) do
				values[i] = tonumber(v)
			end
			local T = table.remove(values, 1)
			local mu = table.remove(values, 1)
			t[T] = t[T] or {}
			t[T][mu] = t[T][mu] or {}
			table.insert(t[T][mu], values)
		end
	end
	for T, u in pairs(t) do
		for mu, v in pairs(u) do
			local w = transpose(v)
			for i = 1, #w do
				w[i] = average(w[i])
			end
			print(T, mu, table.unpack(w))
		end
	end
end

