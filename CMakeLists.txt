cmake_minimum_required(VERSION 3.17)

project(Experiments_AI_TS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS_DEBUG  -g)
SET(EIGEN_DIRECTORY Homing-Pigeon/libs/eigen)

# Includes
include(AddExperiment.cmake)
include(Homing-Pigeon/AddUnitTests.cmake)

# Libraries: 'Homing Pigeon'
add_subdirectory(Homing-Pigeon/libs/hopi)

# Examples of Homing Pigeon
add_experiment(NAME local_minimum_1)
