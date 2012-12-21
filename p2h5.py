#!/usr/bin/python2.7

import pyalps.hdf5 as h5
 
ar = h5.oArchive('test.h5')
ar.write("/parameters/L", 1)
ar.write("/parameters/D", 1)
ar.write("/parameters/T", 0.1)
ar.write("/parameters/g", 0.3)
ar.write("/parameters/mu", 0)
ar.write("/parameters/B", 0.0)
ar.write("/parameters/t", 1)
ar.write("/parameters/J", 0.0)

ar.write("/parameters/THERMALIZATION", 100000)
ar.write("/parameters/SWEEPS", 200000)


