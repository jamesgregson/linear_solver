cmake_minimum_required( VERSION 2.6 )

project( linear_solver )

find_package( GMM REQUIRED )
if( GMM_FOUND )
  include_directories( ${GMM_INCLUDE_DIRS} )
else( GMM_FOUND )
  message( ERROR "GMM++ is required!" )
endif( GMM_FOUND )

find_package( EIGEN3 )
if( EIGEN3_FOUND )
  include_directories( ${EIGEN3_INCLUDE_DIR} )
  add_definitions( -DLINEAR_SOLVER_USES_EIGEN )
else( EIGEN3_FOUND )
  message( STATUS "EIGEN3 Not found, this prevents use of any direct solvers!" )
endif( EIGEN3_FOUND )

find_package( SuperLU )
if( SuperLU_FOUND )
  include_directories( ${SuperLU_INCLUDE_DIRS} )
  add_definitions( -DLINEAR_SOLVER_USES_SUPERLU )
endif( SuperLU_FOUND )


find_package( SuiteSparse REQUIRED )
if( SuiteSparse_FOUND )
  include_directories( ${SuiteSparse_INCLUDE_DIRS} )
  add_definitions( -DLINEAR_SOLVER_USES_CHOLMOD )
endif( SuiteSparse_FOUND )

if( SuiteSparse_FOUND OR SuperLU_FOUND )
  find_package( BLAS REQUIRED )
  include_directories( ${BLAS_INCLUDE_DIRS} )
else( SuiteSparse_FOUND OR SuperLU_FOUND )
  message( ERROR "SuiteSparse and/or SuperLU found, but no BLAS library found!" )
endif( SuiteSparse_FOUND OR SuperLU_FOUND )

set( LIN_SOLVE_HEADERS
  ../include/linear_solver.h
  ../include/linear_solver_utils.h
  ../include/linear_solver_vector.h
  ../include/linear_solver_matrix.h
  ../include/linear_solver_solvers.h
)

set( LIN_SOLVE_SOURCES
)

include_directories( ${CMAKE_SOURCE_DIR}/../include )
add_executable( linear_solver main.cpp ${LIN_SOLVE_HEADERS} ${LIN_SOLVE_SOURCES} )
target_link_libraries( linear_solver ${BLAS_LIBRARIES} ${SuperLU_LIBRARIES} ${SuiteSparse_LIBRARIES} )