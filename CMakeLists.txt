cmake_minimum_required(VERSION 3.16)

project(Experiments_AI_TS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG -g)

# Useful variables
set(HOPI_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/Homing-Pigeon)
set(LIB_CATCH_ROOT ${HOPI_PROJECT_ROOT}/libs/catch)
set(LIB_TORCH_ROOT ${HOPI_PROJECT_ROOT}/libs/torch)
set(EXP_SRCS_SUFFIX "srcs/")

# Includes
list(APPEND CMAKE_MODULE_PATH "${HOPI_PROJECT_ROOT}/cmake")
include(AddExperiment)
include(LinkPyTorchToTarget)

# Library: 'LibTorch'
set(Torch_DIR  "${LIB_TORCH_ROOT}/share/cmake/Torch")
set(Caffe2_DIR "${LIB_TORCH_ROOT}/share/cmake/Caffe2")
find_package(Torch REQUIRED)

# Library: 'Homing Pigeon'
add_subdirectory(${HOPI_PROJECT_ROOT})

# Library: 'Experiments'
set(EXPERIMENTS_SRCS
        # Trackers package
        trackers/TimeTracker.cpp trackers/TimeTracker.h
        trackers/PerformanceTracker.h
        trackers/MazePerformanceTracker.cpp trackers/MazePerformanceTracker.h
        trackers/GraphPerformanceTracker.cpp trackers/GraphPerformanceTracker.h)

# Add prefix to all files in TEST_HOPI_SRCS
list(TRANSFORM EXPERIMENTS_SRCS PREPEND "${EXP_SRCS_SUFFIX}")

create_lib_with_tests(
    LIB
        TARGET experiments
        SOURCES ${EXPERIMENTS_SRCS}
        PUBLIC_LIBS hopi
        PUBLIC_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/srcs
)

# BTAI experiment
add_experiment(NAME main)
