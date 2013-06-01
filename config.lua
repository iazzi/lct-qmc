#!/usr/bin/lua

local function range (a, b, N)
	local i = 0
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

local tasks = setmetatable({}, { __index=table })

local t = 0.2
local U = 4*t
local tx, ty, tz = t, t, t
local J = 4*t*t/U
local L = 4;
local seed = os.time()

--local _, threads = (os.getenv("LSB_HOSTS") or ''):gsub("(%S+)", "%1")
--if threads<1 then threads = 1 end
--print("using "..threads.." threads")
--tasks.THREADS = 48

local mu_min, mu_max = -2*(tx+ty+tz), U/2

for y in range(mu_max, mu_max, 30) do
	for x in range(0.8, 1.1, 20) do
		for _ = 1, 100 do
			seed = seed + 127
			tasks:insert( flip_params{
				Lx = 6,
				Ly = 6,
				Lz = 6,
				T = x*t,
				N = 10/x,
				tx = 1.0*tx,
				ty = 1.0*ty,
				tz = 1.0*tz,
				U = U,
				mu = y,
				B = 0.0,
				h = 1.0e-3,
				THERMALIZATION = 10000,
				SWEEPS = 100000,
				SEED = seed,
				OUTPUT = 'svd_',
				--REWEIGHT = 0,
				SVDPERIOD = 30,
			} )
		end
	end
end

return tasks

