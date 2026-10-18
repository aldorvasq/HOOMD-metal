# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

# Metal GPU platform support.
#
# HOOMD's GPU kernels for Apple GPUs are written in the Metal Shading Language in .metal files. Each
# .metal file is used twice:
#
# * Compiled as C++ into the module: the host driver functions that launch the kernels.
# * Preprocessed into a single Metal source string, checked with the Metal compiler, and embedded
#   in the module. The HIP-on-Metal runtime compiles it when a kernel first launches.

if (ENABLE_HIP AND HIP_PLATFORM STREQUAL "metal")

    if (ENABLE_MPI)
        message(FATAL_ERROR "MPI domain decomposition is not yet available with HOOMD_GPU_PLATFORM=Metal.")
    endif()

    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(IOKIT_FRAMEWORK IOKit REQUIRED)

    set(HOOMD_METAL_DIR ${HOOMD_SOURCE_DIR}/hoomd/mtl)

    # Build-time checker for preprocessed Metal sources
    add_executable(hoomd_mslc ${HOOMD_METAL_DIR}/hoomd_mslc.mm)
    target_compile_options(hoomd_mslc PRIVATE -fobjc-arc)
    target_link_libraries(hoomd_mslc PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

    option(HOOMD_METAL_VALIDATE "Compile Metal kernels at build time to check them" ON)

    # Add .metal kernel files to a target.
    #
    # hoomd_add_metal_sources(<target> <file.metal>...)
    function(hoomd_add_metal_sources target)
        foreach(_src ${ARGN})
            get_filename_component(_abs ${_src} ABSOLUTE)
            get_filename_component(_name ${_src} NAME_WE)
            file(RELATIVE_PATH _rel ${HOOMD_SOURCE_DIR} ${_abs})
            string(MAKE_C_IDENTIFIER "hoomd_mtl_source_${_rel}" _symbol)

            set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/metal)
            set(_out_cc ${_out_dir}/${_name}.metal.cc)
            set(_out_msl ${_out_dir}/${_name}.msl)
            set(_out_dep ${_out_dir}/${_name}.metal.d)

            set(_validate "")
            set(_validate_dep "")
            if (HOOMD_METAL_VALIDATE)
                set(_validate --validator $<TARGET_FILE:hoomd_mslc>)
                set(_validate_dep hoomd_mslc)
            endif()

            set(_defs $<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>)
            set(_incs $<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>)

            add_custom_command(
                OUTPUT ${_out_cc} ${_out_msl}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${_out_dir}
                COMMAND ${Python_EXECUTABLE} ${HOOMD_METAL_DIR}/msl_embed.py
                        --compiler ${CMAKE_CXX_COMPILER}
                        --source ${_abs}
                        --symbol ${_symbol}
                        --output ${_out_cc}
                        --msl ${_out_msl}
                        --depfile ${_out_dep}
                        ${_validate}
                        -I ${HOOMD_METAL_DIR}/msl_stubs
                        -I ${HOOMD_METAL_DIR}/include
                        "$<$<BOOL:${_incs}>:-I$<JOIN:${_incs},;-I>>"
                        "$<$<BOOL:${_defs}>:-D$<JOIN:${_defs},;-D>>"
                DEPENDS ${_abs} ${HOOMD_METAL_DIR}/msl_embed.py ${_validate_dep}
                DEPFILE ${_out_dep}
                COMMAND_EXPAND_LISTS
                COMMENT "Building Metal kernels ${_rel}"
                VERBATIM)

            target_sources(${target} PRIVATE ${_abs} ${_out_cc})
            set_source_files_properties(${_abs} PROPERTIES
                LANGUAGE CXX
                COMPILE_OPTIONS "-xc++"
                COMPILE_DEFINITIONS "HOOMD_MTL_SOURCE=${_symbol}")
        endforeach()
    endfunction()

    # Link stubs for GPU driver functions that are not yet ported to Metal into a target.
    #
    # hoomd_metal_unported_stubs(<target> [<library target>...])
    #
    # Before linking, make_unported_stubs.py finds driver functions that the target references but
    # nothing defines, and builds a static library of stubs that raise an error when called.
    function(hoomd_metal_unported_stubs target)
        set(_archive ${CMAKE_CURRENT_BINARY_DIR}/metal/lib${target}_unported.a)
        set(_libraries "")
        foreach(_lib ${ARGN})
            list(APPEND _libraries $<TARGET_FILE:${_lib}>)
        endforeach()
        add_custom_command(TARGET ${target} PRE_LINK
            COMMAND ${Python_EXECUTABLE} ${HOOMD_METAL_DIR}/make_unported_stubs.py
                    --compiler ${CMAKE_CXX_COMPILER}
                    --output ${_archive}
                    --objects $<TARGET_OBJECTS:${target}>
                    --libraries ${_libraries}
            COMMAND_EXPAND_LISTS
            VERBATIM)
        target_link_libraries(${target} PRIVATE "-Wl,${_archive}")
    endfunction()

endif()

# Configure the Metal version of a templated GPU kernel file.
#
# hoomd_metal_configure_kernel(<template> <output> <list variable>)
#
# When building for Metal and <template>.metal.inc exists, configure it (with @ONLY substitution of
# the caller's variables) to <output>.metal in the current binary directory and append the result
# to <list variable>.
macro(hoomd_metal_configure_kernel _template _output _list)
    if (ENABLE_HIP AND HIP_PLATFORM STREQUAL "metal"
        AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_template}.metal.inc)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_template}.metal.inc
                       ${CMAKE_CURRENT_BINARY_DIR}/${_output}.metal
                       @ONLY)
        list(APPEND ${_list} ${CMAKE_CURRENT_BINARY_DIR}/${_output}.metal)
    endif()
endmacro()

# Collect the Metal ports of a list of .cu files: <name>.metal next to each <name>.cu.
#
# hoomd_metal_ported_sources(<list variable> <file.cu>...)
macro(hoomd_metal_ported_sources _list)
    foreach(_cu ${ARGN})
        string(REGEX REPLACE "\\.cu$" ".metal" _metal ${_cu})
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_metal})
            list(APPEND ${_list} ${CMAKE_CURRENT_SOURCE_DIR}/${_metal})
        endif()
    endforeach()
endmacro()
