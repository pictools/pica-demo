include_directories(
    ${MATHGL_INCLUDE_DIR}
    ${MATHGL_CONFIG_DIR}
    ../../../pica/include
    ../../)

add_executable(demo-baseline
    Main.cpp)

set(DEPS zlib png mgl mgl-static demo-utility)

if (APPLE)
    target_link_libraries(demo-baseline
        -Wl,-all_load
        ${DEPS}
        demo-utility
        -Wl,-noall_load)
elseif (UNIX)
    target_link_libraries(demo-baseline
        -Wl,-whole-archive
        ${DEPS}
        demo-utility
        -Wl,-no-whole-archive)
else()
    target_link_libraries(demo-baseline
        ${DEPS}
        demo-utility)
endif()

add_custom_target(post-demo-baseline ALL DEPENDS demo-baseline
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:demo-baseline>
                               ${ROOT_DIR}/bin/$<TARGET_FILE_NAME:demo-baseline>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mgl>
                               ${ROOT_DIR}/bin/$<TARGET_FILE_NAME:mgl>)

set_target_properties(post-demo-baseline PROPERTIES FOLDER BuildInfrastructure)
