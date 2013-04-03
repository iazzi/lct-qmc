#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local tasks = setmetatable({}, { __index=table })

local t = 0.2
local U = 10*t
local J = 4*t*t/U
local L = 4;
local seed = os.time()

local _, threads = (os.getenv("LSB_HOSTS") or ''):gsub("(%S+)", "%1")
if threads<1 then threads = 1 end
print("using "..threads.." threads")

tasks.THREADS = 1

for x = 0.9, 1.1, 1.05 do
for _, y in ipairs{ 0, } do
	tasks:insert( flip_params{
		Lx = 16,
		Ly = 1,
		Lz = 1,
		T = x*J,
		N = 100/x,
		tx = t,
		ty = 0.14*t,
		tz = 1*t,
		U = U,
		mu = y*t,
		B = 0.0,
		THERMALIZATION = 10000,
		SWEEPS = 100000,
		SEED = seed,
		OUTPUT = 'af6_',
		REWEIGHT = 0,
		LOGFILE = 'af_log',
		DECOMPOSITIONS = 10,
	} )
end
end

return tasks

