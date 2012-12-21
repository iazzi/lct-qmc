#!/usr/bin/python2.7

import pyalps
import matplotlib.pyplot as plt
import pyalps.plot

import xml.etree.ElementTree as ET

import inspect
import numpy

import copy
import glob

exact_params = []
ct_params = []
for L in [ 1, 2, ]:
    for U in [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]:
    #for U in [ 0.0, 0.235, 0.56, 1.1 ]:
	x = {		'LATTICE'                   : "chain lattice",
			'MODEL'                     : "fermion Hubbard",
			'L'                         : L,
			't'			    : 1,
			'J'			    : 0.0,
			'B'			    : 0.0,
			'mu'                        : 0.70,
			'U'                         : -U,
			'CONSERVED_QUANTUMNUMBERS'  : 'Nup,Ndown',
			'MEASURE_AVERAGE[Nup]'      : 'Nup',
			'MEASURE_AVERAGE[Ndown]'    : 'Ndown',
			'THERMALIZATION'            : 10000,
			'SWEEPS'                    : 100000,
			}
	exact_params.append(copy.copy(x))
	for T in [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 2.9, 3.3, 3.7, 4.2 ]:
		x['T'] = T
		ct_params.append(copy.copy(x))


input_file_exact = pyalps.writeInputFiles('exact', exact_params)
input_file_ct = pyalps.writeInputFiles('qmc', ct_params)

pyalps.runApplication('fulldiag', input_file_exact)
pyalps.evaluateFulldiagVersusT(pyalps.getResultFiles(prefix='exact'), DELTA_T=0.02, T_MIN=0.01, T_MAX=4.3)

#plt.figure()
#pyalps.plot.plot(exact_data)
#plt.show()

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

#pyalps.runApplication('../continuoustime', input_file_ct)

run_tasks('../continuoustime', input_file_ct)

#print pyalps.getResultFiles(prefix='qmc')

data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix='qmc'))
plotdata = pyalps.collectXY(sets=data,x='T',y='n_up', foreach=['L', 'U', 'mu'])

measurements = glob.glob('exact.*.measurements.Nup.plot.xml')


for L in [ 1, 2 ]:
	exact_data = []
	for x in measurements:
		exact_data.append(pyalps.readAlpsXMLPlot(x))
	for x in plotdata:
	    if x.props['L']==L:
		exact_data.append(x)
	plt.figure()
	pyalps.plot.plot(exact_data)


plt.show()

