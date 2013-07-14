#!/usr/bin/lua

local function range (a, b, N)
	local i = 0
	if N<0 then a, b, N = b, a, -N end
	if a==b then i, N = 1, 1 end
	return function ()
		if i>N then return nil end
		local ret = a+i*(b-a)/N
		i = i+1
		return ret
	end
end

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local function copy (t)
	local ret = {}
	for k, v in pairs(t) do ret[k] = v end
	return ret
end

local tasks = setmetatable({}, { __index=table })

local t = 0.1
local U = 4*t
local tx, ty, tz = t, t, t
local J = 4*t*t/U
local L = 4;
local seed = 22222 -- os.time()

--local _, threads = (os.getenv("LSB_HOSTS") or ''):gsub("(%S+)", "%1")
--if threads<1 then threads = 1 end
--print("using "..threads.." threads")
--tasks.THREADS = 48

local mu_min, mu_max = -0.5*(tx+ty+tz), U/2

for y in range(mu_max, mu_min, -30) do
	for x in range(0.1, 1.2, 30) do
			seed = seed + 127
			tasks:insert( flip_params{
				Lx = 8,
				Ly = 8,
				Lz = 1,
				T = x*t,
				N = 10/x,
				tx = 1.0*tx,
				ty = 1.0*ty,
				tz = 1.0*tz,
				U = U,
				mu = y,
				B = 0.0,
				h = 0.0e-2,
				THERMALIZATION = 1000,
				SWEEPS = 10000,
				SEED = seed,
				OUTPUT = 'unstable_',
				--REWEIGHT = 0,
				SLICES = 5,
				SVD = 1,
				TIMES = 1,
			} )
	end
end

tasks.dir = "unstable_dir"

return tasks

