cmake_minimum_required(VERSION 3.17)

project(Experiments_AI_TS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS_DEBUG  -g)
SET(HOMING_PIGEON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/Homing-Pigeon)

# Includes
include(AddExperiment.cmake)
include(${HOMING_PIGEON_ROOT}/AddUnitTests.cmake)

# Libraries: 'Homing Pigeon'
add_subdirectory(${HOMING_PIGEON_ROOT}/libs/hopi)

# Examples of Homing Pigeon
add_experiment(NAME local_minimum)
add_experiment(NAME full_vs_partial_expansion)
add_experiment(NAME global_vs_local_inference)
add_experiment(NAME downward_vs_upward_planning)
