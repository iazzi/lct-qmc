Building on Piz Daint
=====================


## Environment
```bash
module switch PrgEnv-cray PrgEnv-gnu
module load boost
module load cray-hdf5
module load cmake
export CRAYPE_LINK_TYPE=dynamic
```


## Build dependencies

### Lua
* Download the sources
* Open the file ```Makefile``` and change the installation prefix, e.g. to ```$(HOME)/apps/daint_gcc/lua```. (Note that the filesystem is shared between multiple machine, it is useful to provide machine specific paths)
* Compile and install with

```bash
make linux
make install
```

### Eigen
* Download the sources and enter in the sources directory
* Simple CMake installation process. Replace ```/path/to/root/install``` with something convenient for you, e.g. ```$(HOME)/apps/daint_gcc/eigen3```

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/root/install ..
make install
```

* Add the pkgconfig directory to your environment. For my example path:

```bash
export PKG_CONFIG_PATH=$HOME/apps/daint_gcc/eigen3/share/pkgconfig:$PKG_CONFIG_PATH
```

### ALPS Core
* Get the latest ```master``` from the repository
* Set the ```BOOST_ROOT``` path (CMake cannot detect it automatically)

```bash
export BOOST_ROOT=/apps/daint/5.2.UP02/boost/1.58.0/gnu_482
```

* Standard CMake build process. I install it in ```$HOME/apps/daint_gcc/alpscore```.

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/root/install ..
make -j10
make install
```

TODO: I think there are still problems with MPI


## Install LCT-AUX
* Set the enviroment variables needed to find the dependencies:

```bash
export LUA_DIR=$HOME/apps/daint_gcc/lua
export PKG_CONFIG_PATH=$HOME/apps/daint_gcc/eigen3/share/pkgconfig:$PKG_CONFIG_PATH
export ALPSCore_DIR=$HOME/apps/daint_gcc/alpscore
export MKLROOT=/opt/intel/15.0.1.133/composer_xe_2015/mkl
```

* Standard CMake build process

```bash
mkdir build && cd build
cmake ..
make -j4
```


## Known issues

### New MKL versions not supported in Eigen
New MKL versions don't define the ```MKL_BLAS``` variable any more. You need to patch the Eigen sources (this is not yeat mainstream as of v3.2.5).

```bash
curl https://bitbucket.org/eigen/eigen/commits/0cc8ceeb7dd1bad74fb975bf8901d5e24090f640/raw/ | git apply -
```

Reference commit: https://bitbucket.org/eigen/eigen/commits/0cc8ceeb7dd1bad74fb975bf8901d5e24090f640




*Michele Dolfi, 15.07.2015*