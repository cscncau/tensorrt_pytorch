cmake_minimum_required(VERSION 3.0)

project(FeatureExtract)

if(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include(CPack)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")#-msse2 ")
  list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-DVERBOSE")
endif()

include_directories (
    ${PROJECT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    /usr/local/cuda/samples/common/inc
    /usr/include/aarch64-linux-gnu
    /usr/src/tensorrt/samples/common
 )
find_package (Threads)
set(sources
   /usr/src/tensorrt/samples/common/logger.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/src/FeatureExtract.cpp
)

link_directories(
  /usr/lib
  /usr/local/cuda/lib64
  /usr/lib/aarch64-linux-gnu
  /usr/lib/aarch64-linux-gnu/tegra
)

set(CUDA_SEPARABLE_COMPILATION ON)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
cuda_add_library(featureextract SHARED ${sources} OPTIONS -arch=sm_62)
set_target_properties(featureextract PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
)

add_definitions("-std=c++11")
target_link_libraries(featureextract
    /usr/local/cuda/lib64/libcudadevrt.a
    -lcudart
    -lcublas
    -lcudnn
    -lnvinfer 
    -lnvparsers 
    -lnvinfer_plugin 
    -lnvonnxparser
)

set(mte_src
   /usr/src/tensorrt/samples/common/logger.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/src/ModelConvertEngine.cpp
)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
cuda_add_executable(model_to_engine ${mte_src} OPTIONS -arch=sm_62)
target_link_libraries(model_to_engine
    /usr/local/cuda/lib64/libcudadevrt.a
    -lcudart
    -lcublas
    -lcudnn
    -lnvinfer 
    -lnvparsers 
    -lnvinfer_plugin 
    -lnvonnxparser
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/
)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
cuda_add_executable(testFeature ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp OPTIONS -arch=sm_62)
target_link_libraries(testFeature
    /usr/local/cuda/lib64/libcudadevrt.a
    -lcudart
    -lcublas
    -lcudnn
    -lfeatureextract
    -lopencv_imgproc
    -lopencv_imgcodecs
    -lopencv_core
)



