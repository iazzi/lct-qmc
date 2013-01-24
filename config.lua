#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local tasks = setmetatable({}, { __index=table })

local t = 0.2
for _, x in ipairs{ 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, } do
	tasks:insert( flip_params{
		L = 10,
		D = 1,
		T = x*t,
		N = 100/x,
		t = t,
		U = 4*t,
		mu = 0.0,
		B = 0.0,
		THERMALIZATION = 10000,
		SWEEPS = 100000,
		OUTPUT = "L10_4.dat",
	} )
end

return table.unpack(tasks)

