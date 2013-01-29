#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local tasks = setmetatable({}, { __index=table })

local t = 0.2
local L = 4;
for x = 0.1, 1.1, 0.1 do
	tasks:insert( flip_params{
		Lx = 4,
		Ly = 4,
		Lz = 1,
		T = x*t,
		N = 100/x,
		tx = t,
		ty = 0.1*t,
		tz = 0.1*t,
		U = 4*t,
		mu = 0.0,
		B = 0.0,
		THREADS = 1,
		THERMALIZATION = 10000,
		SWEEPS = 100000,
		SEED = 42,
		OUTPUT = 'U4_4x4_T'..tostring(x)..'_.dat',
	} )
end

return table.unpack(tasks)

