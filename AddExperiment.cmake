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
    target_include_directories(${ADD_EXPERIMENT_NAME} PUBLIC Homing-Pigeon/libs/eigen)
endfunction()
