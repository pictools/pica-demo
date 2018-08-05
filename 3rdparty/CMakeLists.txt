if (MSVC)
    add_definitions("/wd4996 /wd4800 /wd4334 /wd4267 /wd4251 /wd4244 /wd4190 /wd4018 /wd4217 /wd4305")
endif()

add_subdirectory(zlib)

set(PNG_SHARED OFF CACHE BOOL "Build shared lib" FORCE)
set(PNG_TESTS OFF CACHE BOOL "Build libpng tests" FORCE)
set(SKIP_INSTALL_ALL ON)
set(SKIP_INSTALL_LIBRARIES ON)
add_subdirectory(libpng)

set(enable-opengl OFF CACHE BOOL "Enable OpenGL support" FORCE)
set(enable-lgpl ON CACHE BOOL "Enable only LGPL part of MathGL" FORCE)

add_subdirectory(mathgl)

set_target_properties(zlib PROPERTIES FOLDER 3rdparty)
set_target_properties(png PROPERTIES FOLDER 3rdparty)
set_target_properties(mgl PROPERTIES FOLDER 3rdparty)
set_target_properties(mgl-static PROPERTIES FOLDER 3rdparty)
