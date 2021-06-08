# Function adding an example to the build
function(add_experiment)
    set(options)
    set(args NAME)
    set(list_args)
    cmake_parse_arguments(
            PARSE_ARGV 0
            ADD_EXPERIMENT
            "${options}"
            "${args}"
            "${list_args}"
    )

    foreach(arg IN LISTS ADD_EXPERIMENT_UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed argument: ${arg}")
    endforeach()

    # Add the executable and link the hopi library
    add_executable(${ADD_EXPERIMENT_NAME} experiments/${ADD_EXPERIMENT_NAME}.cpp)
    add_dependencies(${ADD_EXPERIMENT_NAME} hopi)
    target_link_libraries(${ADD_EXPERIMENT_NAME} PUBLIC hopi)

    # Link pytorch
    set(Torch_DIR "${HOMING_PIGEON_ROOT}/libs/torch/share/cmake/Torch")
    set(Caffe2_DIR "${HOMING_PIGEON_ROOT}/libs/torch/share/cmake/Caffe2")

    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_link_libraries(${ADD_EXPERIMENT_NAME} PUBLIC "${TORCH_LIBRARIES}")

    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET hopi
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:hopi>)
    endif (MSVC)
endfunction()
