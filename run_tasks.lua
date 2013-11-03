#!/usr/bin/lua

local name, time = ...
dir = '/cluster/scratch_xl/public/miazzi/'..name..'/'
time = time or '1:00'

local prog = io.popen('pwd'):read('*l')..'/stablefast'
if not io.open(dir..'exec') then
	os.execute('cp '..prog..' '..dir..'exec')
else
end

for i in io.popen('ls '..dir..'/*in'):lines() do
	local t = dofile(i)
	if type(t)=='table' and t[1].outfile then
		local f = io.open(t[1].outfile)
		if not f then
			os.execute('cd '..dir..'; bsub -W '..time..' -J '..name..' ./exec '..i)
		else
			print(t[1].outfile, 'exists')
			f:close()
		end
	else
	end
end
os.execute('bsub -w "ended('..name..')" -o '..dir..name..'.plot lua jk.lua '..dir..'*.dat')
--os.execute('cd '..dir..'; for i in *in;do bsub '..prog..' $i;done')

