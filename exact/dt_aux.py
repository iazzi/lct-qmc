#!/usr/bin/python2.7

import pyalps
import matplotlib.pyplot as plt
import pyalps.plot

import xml.etree.ElementTree as ET

import inspect
import numpy

import copy
import glob

import sys

exact_params = []
ct_params = []

def flip_params (x):
	ret = copy.copy(x)
	ret['U'] = -x['U']
	ret['B'] = 2.0*x['mu'] - x['U']
	ret['mu'] = 0.5*x['B'] - 0.5*x['U']
	return ret

def flip_plot (p, forward = True):
	for i in range(len(p.y)):
		if forward:
			p.y[i] = (1+2*p.y[i])
		else:
			p.y[i] = 0.5*(p.y[i]-1)

cached = False
for a in sys.argv:
	if a=='--cached': cached = True


lengths = [ 6 ]
for L in lengths:
    for U in [ 0.7, ]:
	x = {		'LATTICE'                   : "chain lattice",
			'MODEL'                     : "fermion Hubbard",
			'L'                         : L,
			't'			    : 0.3,
			'J'			    : 0.0,
			'B'			    : 0.0,
			'mu'                        : 0.0,
			'U'                         : U,
			'CONSERVED_QUANTUMNUMBERS'  : 'Nup,Ndown',
			'MEASURE_AVERAGE[N]'      : 'n',
			'MEASURE_AVERAGE[M]'    : 'Sz',
			'THERMALIZATION'            : 10000,
			'SWEEPS'                    : 100000,
			}
	if L<6 :exact_params.append(copy.copy(x))
	for T in [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.6, 3.3, 4.2 ]:
		x['T'] = T*x['t']
		x['dTau'] = 0.01/x['t']
		ct_params.append(flip_params(x))


input_file_exact = pyalps.writeInputFiles('exact', exact_params)
input_file_ct = pyalps.writeInputFiles('qmc', ct_params)

pyalps.runApplication('fulldiag', input_file_exact)
pyalps.evaluateFulldiagVersusT(pyalps.getResultFiles(prefix='exact'), DELTA_T=0.02, T_MIN=0.01, T_MAX=4.3)

def get_tasks (fn):
	ifiles = []
	ofiles = []
	outname = ""
	tree = ET.parse(fn)
	root = tree.getroot()
	assert(root.tag=='JOB')
	for t in root:
		if t.tag=='OUTPUT':
			outname = t.attrib["file"]
		if t.tag=='TASK':
			for x in t:
				if x.tag=='INPUT':
					ifiles.append(x.attrib["file"])
				if x.tag=='OUTPUT':
					ofiles.append(x.attrib["file"])
	return ifiles, ofiles, outname

def run_tasks (appname, fn):
	ifiles, ofiles, outname = get_tasks(fn)
	ret = []
	for i in ifiles:
		ret.append(pyalps.runApplication(appname=appname, parmfile=i))
	return ofiles

if not cached:
	run_tasks('../full', input_file_ct)

data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix='qmc'))
plotdata_N = pyalps.collectXY(sets=data,x='T',y='N', foreach=['L', 'U', 'mu'])
plotdata_M = pyalps.collectXY(sets=data,x='T',y='M', foreach=['L', 'U', 'mu'])


for L in lengths:
	exact_data = []
	measurements = glob.glob('exact.*.measurements.N.plot.xml')
	for x in measurements:
		p = pyalps.readAlpsXMLPlot(x)
		exact_data.append(p)
	measurements = glob.glob('exact.*.measurements.M.plot.xml')
	for x in measurements:
		p = pyalps.readAlpsXMLPlot(x)
		exact_data.append(p)
	for x in plotdata_N:
		flip_plot(x, False)
		x.props['line'] = 'scatter'
	    	if x.props['L']==L:
			exact_data.append(x)
	for x in plotdata_M:
		flip_plot(x, True)
		x.props['line'] = 'scatter'
		if x.props['L']==L:
			exact_data.append(x)
	plt.figure()
	pyalps.plot.plot(exact_data)


plt.show()

