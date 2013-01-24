#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local t = 0.2
local U = 4*t
local T = 0.3*t

return flip_params {
	L = 8,
	D = 1,
	T = T,
	N = 100*t/T,
	t = t,
	U = U,
	mu = 0.0,
	B = 0.0,
	THERMALIZATION = 10000,
	SWEEPS = 100000,
}

