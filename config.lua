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

for x = 0.1, 0.55, 0.05 do
for _, y in ipairs{ -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, } do
	tasks:insert( flip_params{
		Lx = 4,
		Ly = 4,
		Lz = 1,
		T = x*t,
		N = 100/x,
		tx = 1*t,
		ty = 1*t,
		tz = 1*t,
		U = 4*t,
		mu = y*t,
		B = 0.0,
		THREADS = threads,
		THERMALIZATION = 10000,
		SWEEPS = 100000,
		SEED = seed,
		OUTPUT = 'long_',
		REWEIGHT = 0,
		LOGFILE = 'log',
	} )
end
end

return table.unpack(tasks)

