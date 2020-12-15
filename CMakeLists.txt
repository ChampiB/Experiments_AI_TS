cmake_minimum_required(VERSION 3.17)

project(Experiments_AI_TS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS_DEBUG  -g)
SET(HOMING_PIGEON_ROOT Homing-Pigeon)

# Includes
include(AddExperiment.cmake)
include(${HOMING_PIGEON_ROOT}/AddUnitTests.cmake)

# Libraries: 'Homing Pigeon'
add_subdirectory(${HOMING_PIGEON_ROOT}/libs/hopi)

# Examples of Homing Pigeon
add_experiment(NAME local_minimum_1)
