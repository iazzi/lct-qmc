#!/usr/bin/lua

local function flip_params (t)
	t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	return t
end

local tasks = setmetatable({}, { __index=table })

local t = 0.2
local U = 4*t
local tx, ty, tz = t, t, 0.0
local L = 4;
local seed = os.time()

local _, threads = (os.getenv("LSB_HOSTS") or ''):gsub("(%S+)", "%1")
if threads<1 then threads = 1 end

tasks.THREADS = 1
print("using "..tasks.THREADS.." threads")

local mu_min, mu_max = -4*(tx+ty+tz), U/2
local d_mu = (mu_max-mu_min)/30

for x = 0.4, 1.0, 0.025 do
	for y = mu_max, mu_max-6*d_mu, -d_mu do
		for _ = 1, 50 do
			tasks:insert( flip_params{
				Lx = L,
				Ly = L,
				Lz = 1,
				T = x*t,
				N = 100/x,
				tx = tx,
				ty = ty,
				tz = tz,
				U = U,
				mu = y,
				B = 0.0,
				THERMALIZATION = 10000,
				SWEEPS = 100000,
				SEED = seed,
				OUTPUT = 'pl_',
				REWEIGHT = 0,
				DECOMPOSITIONS = 100,
			} )
		end
	end
end

return tasks

