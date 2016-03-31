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


set(PARALLEL_LA 2 CACHE FILEPATH "Use deal.II (0) PETSc(1) or Trilinos(2)?")
add_definitions(-DPARALLEL_LA=${PARALLEL_LA})

set(MAXTHREADS 0 CACHE FILEPATH "Maximum number of threads used per MPI process, 0 means no limit")
add_definitions(-DMAXTHREADS=${MAXTHREADS})

option(CG "Use FE_Q?" OFF)
if(CG)
  add_definitions(-DCG)
endif()

option(PERIODIC "Periodic boundary conditions?" OFF)
if(PERIODIC)
  add_definitions(-DPERIODIC)
endif()

option(MG "Multigrid Preconditioner?" ON)
if (MG)
  add_definitions(-DMG)
endif()

set(SMOOTHENINGSTEPS 10 CACHE FILEPATH "Number of smoothening steps")
add_definitions(-DSMOOTHENINGSTEPS=${SMOOTHENINGSTEPS})

INCLUDE_DIRECTORIES(include)
FILE(GLOB sources source/*.cc)

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  ADD_EXECUTABLE(laplace_meshworker.g ${sources})
  DEAL_II_SETUP_TARGET(laplace_meshworker.g)
ELSEIF(CMAKE_BUILD_TYPE MATCHES "Release")
  ADD_EXECUTABLE(laplace_meshworker ${sources})
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

