LCT-AUX by M. Iazzi, combined with ALPSCore
===========================================

This branch is using ALPSCore for MPI parallelization of the LCT-AUX code
of M. Iazzi (https://gitlab.phys.ethz.ch/iazzi/lct-aux).


## Install

Here we switched to the CMake building framework. The code can be compile running the following:

```bash
mkdir build && cd build
cmake ..
make
make install
```

### Finding dependencies

The code depends on
 * Eigen3
 * Lua
 * ALPSCore
 * (optional) Intel MKL

Here are some variable that can be set when the dependencies are not in the stadard locations.

``Eigen3``
Here we use pkg-config to find the include path. Make sure to have the directory containing ```eigen3.pc``` in the pkg-config path. One can add more search directories for pkg-config with the environment variable ```export PKG_CONFIG_PATH=/path/to/eigen/share/pkgconfig:$PKG_CONFIG_PATH```.


``Lua``
Set the environment variable ```export LUA_DIR=/path/to/lua/root``` to the root of the Lua installation.


``ALPSCore``
Set the environment variable ```export ALPSCore_DIR=/path/to/alpscore/root``` to the root of the ALPSCore installation.


### CMake options
Here are some useful CMake options that can be set during the configuration
```bash
cmake -DOPTION=VALUE ..
```
or later with the CMake interactive interface ```ccmake .```

 * ```ENABLE_MKL``` (ON / OFF) : Whether to enable MKL support in Eigen or not
 * ```CMAKE_BUILD_TYPE``` (Release / Debug / RelWithDebInfo): Which kind of build
 * ```CMAKE_INSTALL_PREFIX``` (PATH) : where to install


