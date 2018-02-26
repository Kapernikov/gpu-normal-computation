set(MODULE_NAME normal-computation)
set(LIBRARY_NAME ${PROJECT_NAME}-${MODULE_NAME})

set(SRCS
    normalComputation.cpp
    covariance_simd.cpp
)

set(DEPENDENCIES
    pcl
    boost
    opencl
   )

add_library(${LIBRARY_NAME} ${SRCS})
target_include_directories(${LIBRARY_NAME} PUBLIC . ${DEPENDENCIES})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${DEPENDENCIES})

add_library(${MODULE_NAME} ALIAS ${LIBRARY_NAME})
