cmake_minimum_required(VERSION 3.14)
project(furthest_point_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── AscendC SDK paths ────────────────────────────────────────────────────────
set(ASCEND_TOOLKIT_HOME "/home/lzt/Ascend/ascend-toolkit/8.3.RC1")
set(ARCH_DIR "${ASCEND_TOOLKIT_HOME}/x86_64-linux")

# Include directories for AscendC headers
# Use ascendc basic_api as the primary set (these include kernel_operator.h etc.)
set(ASCENDC_INCLUDE_DIRS
    # Primary ascendc headers
    ${ARCH_DIR}/ascendc/include/basic_api
    ${ARCH_DIR}/ascendc/include/basic_api/interface
    ${ARCH_DIR}/ascendc/include/basic_api/impl
    ${ARCH_DIR}/ascendc/include/basic_api/impl/utils
    # TikCpp (alternate includes for kernel_operator.h)
    ${ARCH_DIR}/tikcpp/tikcfw
    ${ARCH_DIR}/tikcpp/tikcfw/interface
    ${ARCH_DIR}/tikcpp/tikcfw/interface/micro_api
    ${ARCH_DIR}/tikcpp/tikcfw/impl
    # High-level tiling
    ${ARCH_DIR}/ascendc/include/highlevel_api
    ${ARCH_DIR}/ascendc/include/highlevel_api/kernel_tiling
    # TIK CPU lib (for stub_def.h etc.)
    ${ASCEND_TOOLKIT_HOME}/tools/tikicpulib/lib/include
    # Compiler headers
    ${ARCH_DIR}/ccec_compiler/include
)

# ─── Build target ─────────────────────────────────────────────────────────────
add_executable(furthest_point_sample furthest_point_sample.cpp)

target_compile_definitions(furthest_point_sample PRIVATE
    ASCENDC_CPU_DEBUG=1
    __CCE_AICORE__=310
    __NPU_ARCH__=3102
)

target_include_directories(furthest_point_sample PRIVATE ${ASCENDC_INCLUDE_DIRS})

# TIK CPU debug library (provides half implementations etc.)
set(TIKCPULIB_DIR "${ASCEND_TOOLKIT_HOME}/tools/tikicpulib/lib")
target_link_directories(furthest_point_sample PRIVATE
    ${TIKCPULIB_DIR}
    ${TIKCPULIB_DIR}/Ascend310P1
    ${ARCH_DIR}/simulator/Ascend310P1/lib
    ${ARCH_DIR}/devlib
    ${ARCH_DIR}/lib64
)
target_link_libraries(furthest_point_sample PRIVATE
    tikcpp_debug
    _pvmodel
    tikicpulib_stubreg
    pthread dl
)

target_compile_options(furthest_point_sample PRIVATE -O2 -g -Wall -fPIC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
