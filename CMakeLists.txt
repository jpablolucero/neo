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

INCLUDE_DIRECTORIES(include include/local_integrators)
FILE(GLOB sources source/*.cc)

ADD_EXECUTABLE(laplace_meshworker ${sources})
DEAL_II_SETUP_TARGET(laplace_meshworker)
