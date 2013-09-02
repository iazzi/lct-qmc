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

local function average (t, w)
	local sum = 0
	local n = 0
	if w then
		for k, v in pairs(t) do
			sum, n = sum + v/w[k], n + 1.0/w[k]
		end
	else
		for k, v in pairs(t) do
			sum, n = sum + v, n + 1
		end
	end
	return sum / n
end

local function jk (t)
	local sum = 0
	local sum2 = 0.0
	local n = 0
	for k, v in pairs(t) do
		n = n + 1
		sum = sum + v
		sum2 = sum2 + v*v
	end
	for k, v in pairs(t) do
	end
	return math.sqrt(sum2/n - sum*sum/n/n)/math.sqrt(n)
end

local files = {}
for _, fn in ipairs{...} do
	local p = io.popen('ls '..fn)
	for f in p:lines() do
		table.insert(files, f)
		--print(f)
	end
	p:close()
end
local out = {}
for _, fn in ipairs(files) do
	--print("doing", fn)
	local t = {}
	local f = assert(io.open(fn))
	for l in f:lines() do
		if l:match('^%s*$') or l:match('%#.*') then
		else
			local values = {}
			for m in l:gmatch('(%S+)') do
				table.insert(values, m)
			end
			for i, v in ipairs(values) do
				values[i] = tonumber(v) or 0.0
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
			for i = 1, #w, 2 do
				local e = jk(w[i])
				w[i] = average(w[i])
				w[i+1] = math.sqrt(average(w[i+1])/#w[i+1])
				--w[i+1] = math.sqrt(w[i+1]-w[i]*w[i])
			end
			table.insert(out, { T, mu, table.unpack(w) })
			--print(T, mu, table.unpack(w))
		end
	end
end
table.sort(out, function(x, y) if x[2]<y[2] then return true elseif x[2]==y[2] and x[1]<y[1] then return true else return false end end)
local T
for _, t in ipairs(out) do
	if t[2]~=T then print() T = t[2] end
	print(table.unpack(t))
end

