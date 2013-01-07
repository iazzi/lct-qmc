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

r_params = []
a_params = []

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


lengths = [ 1, 2 ]
for L in lengths:
    for U in [ 0.5 ]:
	x = {		'LATTICE'                   : 'chain lattice',
			'MODEL'                     : 'fermion Hubbard',
			'L'                         : L,
			't'			    : 1,
			'J'			    : 0.0,
			'B'			    : 1.0,
			'mu'                        : -1.0,
			'U'                         : U,
			'CONSERVED_QUANTUMNUMBERS'  : 'Nup,Ndown',
			'MEASURE_AVERAGE[N]'        : 'n',
			'MEASURE_AVERAGE[M]'        : 'Sz',
			'THERMALIZATION'            : 10000,
			'SWEEPS'                    : 100000,
			}
	r_params.append(copy.copy(x))
	a_params.append(flip_params(x))

r_input_file = pyalps.writeInputFiles('rep', r_params)
a_input_file = pyalps.writeInputFiles('att', a_params)

pyalps.runApplication('fulldiag', r_input_file)
pyalps.runApplication('fulldiag', a_input_file)

pyalps.evaluateFulldiagVersusT(pyalps.getResultFiles(prefix='rep'), DELTA_T=0.02, T_MIN=0.01, T_MAX=4.3)
pyalps.evaluateFulldiagVersusT(pyalps.getResultFiles(prefix='att'), DELTA_T=0.02, T_MIN=0.01, T_MAX=4.3)

measurements = glob.glob('rep.*.measurements.N.plot.xml')
for m in measurements:
	data = []
	p = pyalps.readAlpsXMLPlot(m)
	p.props['title'] = m
	data.append(p)
	m2 = 'att' + m[3:-10] + 'M.plot.xml'
	p = pyalps.readAlpsXMLPlot(m2)
	#p.props['title'] = m2
	p.props['line'] = 'scatter'
	flip_plot(p, True)
	data.append(p)
	plt.figure()
	pyalps.plot.plot(data)

measurements = glob.glob('rep.*.measurements.M.plot.xml')
for m in measurements:
	data = []
	p = pyalps.readAlpsXMLPlot(m)
	p.props['title'] = m
	data.append(p)
	m2 = 'att' + m[3:-10] + 'N.plot.xml'
	p = pyalps.readAlpsXMLPlot(m2)
	#p.props['title'] = m2
	p.props['line'] = 'scatter'
	flip_plot(p, False)
	data.append(p)
	plt.figure()
	pyalps.plot.plot(data)

plt.show()

