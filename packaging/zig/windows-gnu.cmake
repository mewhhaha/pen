# The CMake toolchain for C dependencies (libopus, through opus-head-sys)
# when cargo-zigbuild cross-compiles pen to x86_64-pc-windows-gnu; the
# Justfile's `windows` recipe sets CMAKE_TOOLCHAIN_FILE_x86_64_pc_windows_gnu
# to it, which cargo-zigbuild then leaves alone. It is cargo-zigbuild's own
# toolchain but for the compiler, which drops MSVC's /guard:cf: opus-head-sys
# passes it for every Windows target, and zig's clang driver rejects it.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_C_COMPILER ${CMAKE_CURRENT_LIST_DIR}/windows-gnu-cc.sh)
set(CMAKE_AR ${CMAKE_CURRENT_LIST_DIR}/ar.sh)
set(CMAKE_RANLIB ${CMAKE_CURRENT_LIST_DIR}/ranlib.sh)
set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
