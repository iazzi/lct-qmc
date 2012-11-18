#!/usr/bin/python2.7

import pyalps.hdf5 as h5
 
ar = h5.oArchive('test.h5')
ar.write("/parameters/L", 4)
ar.write("/parameters/D", 2)
ar.write("/parameters/T", 0.1)
ar.write("/parameters/g", 0.5)
ar.write("/parameters/mu", -1.5)
ar.write("/parameters/B", 0.0)
ar.write("/parameters/t", 1.0)
ar.write("/parameters/J", 0.0)

ar.write("/parameters/THERMALIZATION", 100000)
ar.write("/parameters/SWEEPS", 100000)


