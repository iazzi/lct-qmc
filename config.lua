#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local tasks = setmetatable({}, { __index=table })

local t = 0.3
local U = 4*t
local tx, ty, tz = t, t, t
local J = 4*t*t/U
local L = 4;
local seed = os.time()

local _, threads = (os.getenv("LSB_HOSTS") or ''):gsub("(%S+)", "%1")
if threads<1 then threads = 1 end
print("using "..threads.." threads")

tasks.THREADS = threads

local mu_min, mu_max = -2*(tx+ty+tz), U/2
local d_mu = (mu_max-mu_min)/30

for y = mu_max, mu_max, -d_mu do
	for x = 1.4, 0.8, -0.05 do
		for _ = 1, 50 do
			seed = seed + 127
			tasks:insert( flip_params{
				Lx = 8,
				Ly = 8,
				Lz = 2,
				T = x*t,
				N = 10/x,
				tx = tx,
				ty = ty,
				tz = tz,
				U = U,
				mu = y,
				B = 0.0,
				h = 1.0e-3,
				THERMALIZATION = 10000,
				SWEEPS = 100000,
				SEED = seed,
				OUTPUT = 's_p_',
				--REWEIGHT = 0,
				DECOMPOSITIONS = 100,
			} )
		end
	end
end

return tasks

