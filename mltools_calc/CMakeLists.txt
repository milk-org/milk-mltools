# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.16)
project(lib_mltools_calc_project C)

find_package(OpenMP REQUIRED)

add_library(mltools_calc SHARED mlt_calc_ext.c)

target_include_directories(
  mltools_calc PRIVATE "${CMAKE_SOURCE_DIR}/src/calc"
                       "${CMAKE_SOURCE_DIR}/src/engine/libfps"
                       "${CMAKE_SOURCE_DIR}/src/engine")

target_compile_options(
  mltools_calc PRIVATE -O3 -march=native -fopenmp -fopenmp-simd
                       -fno-finite-math-only -Wall -Wextra -Wno-unused-parameter)

target_link_libraries(mltools_calc PRIVATE OpenMP::OpenMP_C m)

install(TARGETS mltools_calc LIBRARY DESTINATION lib/milk/calc.d)
