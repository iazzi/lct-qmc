# - FindMKL.cmake
# Find LAPACK/BLAS (and compatible) numerical libraries
#  Copyright (C)  2012-2015 Michele Dolfi <dolfim@phys.ethz.ch>
#  Copyright (C)  2009-2012 Ryo IGARASHI <rigarash@issp.u-tokyo.ac.jp>
#
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)
#
########################################################################################################
# Ouput:
#  MKL_FOUND               - was MKL found
#  MKL_INCLUDE_DIRS        - path to the MKL include files
#  MKL_LIBRARIES           - libraries to use MKL
########################################################################################################
# Looking for MKL.
#   For parallel MKL, OpenMP check has to be done beforehand.
# 0) $ENV{MKL} can be defined from http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
#    if specified, this settings are chosen
# 1) If compiler is Intel >= 12 (Intel Composer XE 2011/2013, Intel Compiler Pro)
# 1.1) When OPENMP_FOUND=ON and USE_MKL_PARALLEL=ON, use -mkl=parallel
# 1.2) When OPENMP_FOUND=OFF or USE_MKL_PARALLEL=OFF, use -mkl=sequential
# 2) If $ENV{MKLROOT} / $ENV{MKL_HOME} defined (done by MKL tools/environment scripts), use the linking from advisor
# 3) Look for MKL libraries in MKL_PATHS
########################################################################################################

SET(MKL_PATHS "/usr/local/lib /usr/lib")

# 0) $ENV{MKL} can be set for explicit linking
if($ENV{MKL} MATCHES "mkl")
  set(MKL_LIBRARIES $ENV{MKL})
endif($ENV{MKL} MATCHES "mkl")

# 1) Intel compiler >= 12
if(NOT MKL_LIBRARIES)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(INTEL_WITH_MKL FALSE)
    if(DEFINED CMAKE_CXX_COMPILER_VERSION)
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 12
         OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 12)
        set(INTEL_WITH_MKL TRUE)
      endif()
    endif()
    if($ENV{MKLROOT} MATCHES "composer")
      set(INTEL_WITH_MKL TRUE)
    endif()
    
    if(INTEL_WITH_MKL)
      if(OPENMP_FOUND AND USE_MKL_PARALLEL)
        set(MKL_LIBRARIES "-mkl=parallel")
      else(OPENMP_FOUND AND USE_MKL_PARALLEL)
        set(MKL_LIBRARIES "-mkl=sequential")
      endif(OPENMP_FOUND AND USE_MKL_PARALLEL)
      set(HAVE_MKL TRUE)
    endif(INTEL_WITH_MKL)
  endif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
endif(NOT MKL_LIBRARIES)

# 2) Use MKLROOT / MKL_HOME and standard linking
if(NOT MKL_LIBRARIES)
  set(mkl_home "")
  if($ENV{MKLROOT} MATCHES "mkl")
    set(mkl_home $ENV{MKLROOT})
  elseif($ENV{MKL_HOME} MATCHES "mkl")
    set(mkl_home $ENV{MKL_HOME})
  endif()

  if(mkl_home MATCHES "mkl")
    file( STRINGS "${mkl_home}/include/mkl.h" _mkl_h_content REGEX "__INTEL_MKL" )
    string(REGEX REPLACE ".*#define __INTEL_MKL__ ([0-9]+).*"        "\\1" MKL_VERSION_MAJOR  "${_mkl_h_content}")
    string(REGEX REPLACE ".*#define __INTEL_MKL_MINOR__ ([0-9]+).*"  "\\1" MKL_VERSION_MINOR  "${_mkl_h_content}")
    string(REGEX REPLACE ".*#define __INTEL_MKL_UPDATE__ ([0-9]+).*" "\\1" MKL_VERSION_UPDATE "${_mkl_h_content}")
    set(MKL_VERSION "${MKL_VERSION_MAJOR}.${MKL_VERSION_MINOR}.${MKL_VERSION_UPDATE}")
    
    # message(STATUS "LAPACK DEBUG::Compiler id ${CMAKE_CXX_COMPILER_ID}")
    # message(STATUS "LAPACK DEBUG::Compiler version ${CMAKE_CXX_COMPILER_VERSION}")
    # message(STATUS "LAPACK DEBUG::Processor ${CMAKE_SYSTEM_PROCESSOR}")
    # message(STATUS "LAPACK DEBUG::ENV{MKL} $ENV{MKL}")
    # message(STATUS "LAPACK DEBUG::ENV{MKLROOT} $ENV{MKLROOT}")
    # message(STATUS "LAPACK DEBUG::ENV{MKL_HOME} $ENV{MKL_HOME}")
    # message(STATUS "LAPACK DEBUG::MKL_VERSION ${MKL_VERSION}")
    
    # OS thread library (pthread required by MKL)
    find_package(Threads REQUIRED)
    # MKL core
    if(OPENMP_FOUND AND ALPS_USE_MKL_PARALLEL)
      # No parallel mode support for MKL < 10.0
      if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        # Intel with Intel OpenMP
        set(MKL_CORE -lmkl_intel_thread -lmkl_core -liomp5)
      elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        # GCC with GNU OpenMP
        # MKL with g++ needs gfortran
        set(MKL_CORE -lmkl_gnu_thread -lmkl_core -lgfortran)
      endif()
    else()
      if(${MKL_VERSION} MATCHES "1[0-1]\\.[0-3]\\.[0-9]+")
        set(MKL_CORE -lmkl_sequential -lmkl_core)
      else() # MKL < 10.0
        set(MKL_CORE -lmkl_lapack -lmkl -lguide)
      endif()
    endif()
    # basic data type model interface
    # - assuming ILP32 or LP64
    if(${MKL_VERSION} MATCHES "1[0-1]\\.[0-3]\\.[0-9]+")
      if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "ia64")
        set(MKL_INTERFACE -lmkl_intel_lp64)
      elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "i386" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "i686")
        set(MKL_INTERFACE -lmkl_intel)
      else()
        message(SEND_ERROR "MKL: the processor type of this system is not supported")
      endif()
    else() # MKL < 10.0
      set(MKL_INTERFACE "")
    endif()
    # MKL library path
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      if(${MKL_VERSION} MATCHES "11\\.[0-9]\\.[0-9]+" OR ${MKL_VERSION} MATCHES "10\\.3\\.[0-9]+")
        set(MKL_LIBRARY_PATH -L${mkl_home}/lib)
      else() # MKL < 10.3
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/em64t)
        elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "i386" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "i686")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/32)
        else()
          message(SEND_ERROR "MKL: the processor type of this system is not supported")
        endif()
      endif()
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      if(${MKL_VERSION} MATCHES "11\\.[0-9]\\.[0-9]+" OR ${MKL_VERSION} MATCHES "10\\.3\\.[0-9]+")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/intel64)
        elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "i386" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "i686")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/ia32)
        else()
          message(SEND_ERROR "MKL: the processor type of this system is not supported")
        endif()
      else() # MKL < 10.3 have the same PATH
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/em64t)
        elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "i386" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "i686")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/32)
        elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ia64")
          set(MKL_LIBRARY_PATH -L${mkl_home}/lib/64)
        else()
          message(SEND_ERROR "MKL: the processor type of this system is not supported")
        endif()
      endif()
    endif()
    # combine together
    set(MKL_LIBRARIES ${MKL_LIBRARY_PATH} ${MKL_INTERFACE} ${MKL_CORE} ${CMAKE_THREAD_LIBS_INIT} -lm)

    # unset local variables
    unset(MKL_LIBRARY_PATH)
    unset(MKL_INTERFACE)
    unset(MKL_CORE)

  endif(mkl_home MATCHES "mkl")
endif(NOT MKL_LIBRARIES)

IF(MKL_LIBRARIES)
  include(CheckFunctionExists)
  # Checking if it works
  set(CMAKE_REQUIRED_LIBRARIES ${MKL_LIBRARIES})
  check_function_exists("sgemm" _libraries_work)
  set(CMAKE_REQUIRED_LIBRARIES)
  
  if(NOT _libraries_work)
    message(WARNING "MKL was detected but I'm not able to use it.")
    message(STATUS "MKL settings were:")
    message(STATUS "   BLAS_LIBRARY = ${BLAS_LIBRARY}")
    message(STATUS "   LAPACK_LIBRARY = ${LAPACK_LIBRARY}")
    set(MKL_LIBRARIES)
  else()
    message(STATUS "Found intel/mkl library")
    set(MKL_INC_PATHS ${mkl_home}/include ${MKL_PATHS}) 
    find_path(MKL_INCLUDE_DIRS mkl.h ${MKL_INC_PATHS})
  endif(NOT _libraries_work)
ENDIF(MKL_LIBRARIES)


# include this to handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL REQUIRED_VARS MKL_LIBRARIES MKL_INCLUDE_DIRS)
