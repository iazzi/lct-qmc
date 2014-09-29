#!/usr/bin/lua

local function write (out, beta, U, mu, K, name, time)
out:write((([[
#!/usr/local/bin/bash
#SBATCH --partition=dphys_compute
#SBATCH --nodes=1
#SBATCH --job-name=%NAME.job
#SBATCH --time=%TIME
#SBATCH --ntasks=20
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpu_bind=verbose
#SBATCH --output=/mnt/lnec/iazzi/%NAME/beta_%B_U_%U.out
#SBATCH --error=/mnt/lnec/iazzi/%NAME/beta_%B_U_%U.err

srun --signal=14@60 /mnt/lnec/iazzi/]]..name..[[/v3ct %B %M %U %K /mnt/lnec/iazzi/%NAME/beta_%B_U_%U.%J.dat
]]):gsub('%%(%u+)', {
	J = '%J',
	B = beta,
	U = U,
	M = mu,
	K = K,
	NAME = name,
	TIME = time,
})))
end


local name = ...
local n = 1
for beta = 1, 10 do
	local f = io.open(name..'_'..n..'.in', 'w')
	local time = 1000 * beta * beta
	time = ('%.2i:%.2i:%.2i'):format(math.floor(time/3600), math.floor(time/60)%60, time%60)
	write(f, beta, 2, 1, 5, name, time)
	f:close()
	n = n + 1
end
os.execute('mkdir /mnt/lnec/iazzi/'..name)
os.execute('cp /users/iazzi/bss-mc/v3ct /mnt/lnec/iazzi/'..name..'/v3ct')

