# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later

set(LIBNAME "milkmltools")
set(SRCNAME "mltools")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

find_package(OpenMP REQUIRED)

file(GLOB SOURCEFILES "*.c")
file(GLOB INCLUDEFILES "*.h")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project C)

milk_add_runnable_module(
  ${LIBNAME}
  SOURCES ${SOURCEFILES}
  LINK_LIBS $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits_static>)
milk_add_fpsexec_stubs(${LIBNAME})

target_include_directories(
  ${LIBNAME}_objs PRIVATE "${CMAKE_SOURCE_DIR}/src"
                           "${CMAKE_SOURCE_DIR}/src/coremods")
target_compile_options(
  ${LIBNAME}_objs PRIVATE -O3 -fopenmp -fopenmp-simd -Wall -Wextra -Wformat=2
                          -Wformat-overflow -Wformat-truncation)

target_link_libraries(
  ${LIBNAME} PUBLIC OpenMP::OpenMP_C
                    $<$<AND:$<BOOL:${USE_CFITSIO}>,$<BOOL:${CFITSIO_FOUND}>>:milkCOREMODiofits>)

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(FILES commands.d/mltools.bash DESTINATION share/milk/commands.d)
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

include(CTest)
add_test(NAME mlt-tlmindex-h1 COMMAND milk-fpsexec-mlt-tlmindex -h1)
add_test(NAME mlt-tlmload-h1 COMMAND milk-fpsexec-mlt-tlmload -h1)
add_test(NAME mlt-tlmsync-h1 COMMAND milk-fpsexec-mlt-tlmsync -h1)
add_test(NAME mlt-tlmsave-h1 COMMAND milk-fpsexec-mlt-tlmsave -h1)
add_test(NAME mlt-aoopt-h1 COMMAND milk-fpsexec-mlt-aoopt -h1)
add_test(NAME mlt-psfrecon-h1 COMMAND milk-fpsexec-mlt-psfrecon -h1)
