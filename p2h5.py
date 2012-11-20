#!/usr/bin/python2.7

import pyalps.hdf5 as h5
 
ar = h5.oArchive('test.h5')
ar.write("/parameters/L", 2)
ar.write("/parameters/D", 2)
ar.write("/parameters/T", 0.01)
ar.write("/parameters/g", 0.4)
ar.write("/parameters/mu", 0.0)
ar.write("/parameters/B", 0.0)
ar.write("/parameters/t", 0.1)
ar.write("/parameters/J", 0.0)

ar.write("/parameters/THERMALIZATION", 100000)
ar.write("/parameters/SWEEPS", 100000)


