#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local tasks = setmetatable({}, { __index=table })

local t = 0.2
local L = 4;
local seed = 42
local seed = os.time()

local _, threads = (os.getenv("LSB_HOSTS") or ''):gsub("(%S+)", "%1")
if threads<1 then threads = 1 end
print("using "..threads.." threads")

for x = 0.1, 1.1, 0.1 do
	tasks:insert( flip_params{
		Lx = 8,
		Ly = 1,
		Lz = 1,
		T = x*t,
		N = 100/x,
		tx = 1*t,
		ty = 1*t,
		tz = 1*t,
		U = 4*t,
		mu = 0.0,
		B = 0.0,
		THREADS = threads,
		THERMALIZATION = 10000,
		SWEEPS = 100000,
		SEED = seed,
		OUTPUT = 'last_',
		RESET = true,
	} )
end

return table.unpack(tasks)

