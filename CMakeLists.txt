CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS $ENV{DEAL_II_DIR} ${DEAL_II_DIR} $ENV{DEAL_DIR} ${DEAL_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(laplace_meshworker)

SET(PARALLEL_LA 2 CACHE STRING "Use deal.II (0) PETSc(1) or Trilinos(2)?")
ADD_DEFINITIONS(-DPARALLEL_LA=${PARALLEL_LA})

SET(MAXTHREADS 0 CACHE STRING "Maximum number of threads used per MPI process, 0 means no limit")
ADD_DEFINITIONS(-DMAXTHREADS=${MAXTHREADS})

OPTION(CG "Use FE_Q?" OFF)
IF(CG)
  ADD_DEFINITIONS(-DCG)
ENDIF()

OPTION(PERIODIC "Periodic boundary conditions?" OFF)
IF(PERIODIC)
  ADD_DEFINITIONS(-DPERIODIC)
ENDIF()

OPTION(MG "Multigrid Preconditioner?" ON)
IF (MG)
  ADD_DEFINITIONS(-DMG)
ENDIF()

OPTION(BENCHMARKS "Benchmarks?" OFF)
IF (BENCHMARKS)
  ADD_DEFINITIONS(-DBENCHMARKS)
  SET(GOOGLE_BENCHMARK_PATH "" CACHE PATH "Location of the install path of google benchmark")
  ADD_DEFINITIONS(-DGOOGLE_BENCHMARK_PATH=${GOOGLE_BENCHMARK_PATH})
  INCLUDE_DIRECTORIES(${GOOGLE_BENCHMARK_PATH}/include)
  FIND_PACKAGE(Threads REQUIRED)
  ADD_LIBRARY(benchmark STATIC IMPORTED)
  SET_TARGET_PROPERTIES(benchmark PROPERTIES
    IMPORTED_LOCATION 
    ${GOOGLE_BENCHMARK_PATH}/lib/libbenchmark.a)
ENDIF()

OPTION(HEADER_IMPLEMENTATION "Implement source in headers" OFF)
IF (HEADER_IMPLEMENTATION)
  ADD_DEFINITIONS(-DHEADER_IMPLEMENTATION)
  INCLUDE_DIRECTORIES(source)
ENDIF()

IF(HEADER_IMPLEMENTATION)
  IF (BENCHMARKS)
    FILE(GLOB sources source/Benchmarks.cc)
  ELSE()
    FILE(GLOB sources source/LaplaceMeshworker.cc)
  ENDIF()
ELSE()
  FILE(GLOB sources source/*.cc)
ENDIF()

OPTION(FLTO "Link time optimization" OFF)
IF (FLTO)
  ADD_DEFINITIONS(-DFLTO)
  SET(CMAKE_EXE_LINKER_FLAGS  " ${CMAKE_EXE_LINKER_FLAGS} -flto ") 
ENDIF()

INCLUDE_DIRECTORIES(include include/local_integrators)

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  ADD_EXECUTABLE(laplace_meshworker.g ${sources})
  IF (BENCHMARKS)
    TARGET_LINK_LIBRARIES(laplace_meshworker.g benchmark ${CMAKE_THREAD_LIBS_INIT})
  ENDIF()
  DEAL_II_SETUP_TARGET(laplace_meshworker.g)
ELSEIF(CMAKE_BUILD_TYPE MATCHES "Release")
  ADD_EXECUTABLE(laplace_meshworker ${sources})
  IF (BENCHMARKS)
    TARGET_LINK_LIBRARIES(laplace_meshworker benchmark ${CMAKE_THREAD_LIBS_INIT})
  ENDIF()
  DEAL_II_SETUP_TARGET(laplace_meshworker)
ELSE()
  MESSAGE(FATAL_ERROR
    "CMAKE_BUILD_TYPE does neither match Release nor Debug!"
    )
ENDIF()

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

ADD_CUSTOM_TARGET(both)
ADD_DEPENDENCIES(both debug release)

