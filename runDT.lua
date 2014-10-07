#!/usr/bin/lua

local function write (out, a, b, name, time)
	local bstring = [[
#!/usr/local/bin/bash
#SBATCH --partition=dphys_compute
#SBATCH --nodes=1
#SBATCH --job-name=%NAME.job
#SBATCH --time=%TIME
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpu_bind=verbose
#SBATCH --output=/mnt/lnec/iazzi/%NAME/]]..a..'_'..b..[[.out
#SBATCH --error=/mnt/lnec/iazzi/%NAME/]]..a..'_'..b..[[.err

cd ]]..'/mnt/lnec/iazzi/'..name
	for i = a, b do
		bstring = bstring .. '/mnt/lnec/iazzi/'..name..'/main /mnt/lnec/iazzi/%NAME/'..i..'.in &\n'
	end
	out:write(bstring:gsub('%%NAME', name):gsub('%%TIME', time), 'wait\n')
end


local name, time = ...
os.execute('mkdir /mnt/lnec/iazzi/'..name)
os.execute('cp /users/iazzi/bss-mc/main /mnt/lnec/iazzi/'..name..'/main')
for n = 1, 1000, 20 do
	local t = io.open('/mnt/lnec/iazzi/'..name..'/'..n..'.in')
	if not t then break end
	t:close()
	local f = io.open('/mnt/lnec/iazzi/'..name..'/'..n..'_'..(n+19)..'.batch', 'w')
	write(f, n, n+19, name, time)
	f:close()
	os.execute('sbatch /mnt/lnec/iazzi/'..name..'/'..n..'_'..(n+19)..'.batch')
end

