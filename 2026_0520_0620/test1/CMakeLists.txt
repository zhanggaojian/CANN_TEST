cmake_minimum_required(VERSION 3.16)
project(celu_custom_op LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        message(FATAL_ERROR "Set ASCEND_CANN_PACKAGE_PATH or ASCEND_HOME_PATH before building.")
    endif()
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

add_library(celu_host SHARED
    op_host/celu.cpp
)

target_link_directories(celu_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(celu_host PRIVATE
    register
    tiling_api
    platform
)

add_library(celu_kernel OBJECT
    op_kernel/celu.cpp
)

target_include_directories(celu_kernel PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

