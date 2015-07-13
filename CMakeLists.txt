# we need at least cmake version 2.8.12
cmake_minimum_required(VERSION 2.8.12)

# the project is called 'linking', and it is a C++ project
project(lct-aux CXX C)

# Some config options: EIGEN_NO_DEBUG, C++11
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "-Wno-unknown-pragmas -DNDEBUG -DEIGEN_NO_DEBUG ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "CMAKE_CXX_FLAGS_RELEASE: Flags used by the compiler during release builds" FORCE)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Wno-unknown-pragmas -DNDEBUG -DEIGEN_NO_DEBUG ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "CMAKE_CXX_FLAGS_RELWITHDEBINFO: Flags used by the compiler during Release with Debug Info builds." FORCE)
  set(CMAKE_C_FLAGS_RELEASE " ${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "CMAKE_C_FLAGS_RELEASE: Flags used by the compiler during release builds" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Add custom FindXX.cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)



set(MYTARGET_LIBRARIES "")

# the project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore REQUIRED)
list(APPEND MYTARGET_LIBRARIES ${ALPSCore_LIBRARIES})

# if Lua is not found automatically,
# define LUA_DIR in ENV
find_package(Lua REQUIRED)
include_directories(${LUA_INCLUDE_DIR})
list(APPEND MYTARGET_LIBRARIES ${ALPSCore_LIBRARIES})

# Fing Eigen3 through pkg-config
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

# Find MKL
option(ENABLE_MKL "Enable the use of MKL with Eigen" ON)
if(ENABLE_MKL)
  find_package( MKL REQUIRED )
  add_definitions(-DEIGEN_USE_MKL_ALL)
  include_directories( ${MKL_INCLUDE_DIRS} )
  list(APPEND MYTARGET_LIBRARIES ${MKL_LIBRARIES})
endif()


## Targets
add_executable(generic generic.cpp)
target_link_libraries(generic ${MYTARGET_LIBRARIES})

add_executable(generic_alps generic_alps.cpp lctaux_sim.cpp)
target_link_libraries(generic_alps ${MYTARGET_LIBRARIES})

add_executable(generic_mpi generic_mpi.cpp lctaux_sim.cpp)
target_link_libraries(generic_mpi ${MYTARGET_LIBRARIES})

