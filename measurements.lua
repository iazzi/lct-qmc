#!/usr/bin/lua

local function mean (m, i)
	i = i or 1
	return m.sums[i]/m.samples[i]
end

local function variance (m, i)
	i = i or 1
	return m.squares[i]/m.samples[i] - mean(m, i)*mean(m, i)
end

local function err (m, i)
	i = i or 1
	return math.sqrt(variance(m, i)/m.samples[i])
end

local function time (m, i)
	i = i or 1
	return (m:variance(i)*m.samples[1]/m.samples[i]/m:variance(1)-1.0)*0.5;
end

local function measurement_string_short (m)
	if m.bins==nil or m.bins<1 then return (m.name or 'Result')..": empty" end
	local N = math.max(1, m.bins-5)
	return (m.name or 'Result')..": "..mean(m, N).." +- "..m:error(N).." tau="..m:time(N)
end

local function add (m, x, i)
	local nx = x;
	i = i or 1
	while true do
		if i>m.bins then
			m.sums[i] = nx
			m.squares[i] = nx*nx
			m.values[i] = nx
			m.samples[i] = 1
			m.bins = m.bins+1
		else
			m.sums[i] = m.sums[i] + nx
			m.squares[i] =m.squares[i] + nx*nx
			m.values[i] = nx
			m.samples[i] = m.samples[i] + 1;
		end
		if m.samples[i]%2==1 then break end
		nx = (m.values[i]+nx)/2.0;
		i = i + 1
	end
end

local measurement_index = {
	mean = mean,
	variance = variance,
	time = time,
	error = err,
	add = add,
	tostring = measurement_string_short,
}

local measurement_mt = {
	__index = measurement_index,
	__tostring = measurement_string_short,
}

return {
	new = function(t) return setmetatable(t or {
		bins = 0,
		sums = {},
		squares = {},
		values = {},
		samples = {},
	}, measurement_mt) end
}
