include_directories(
    ../
    ../../pica/include
    ../../3rdparty)

add_library(demo-utility
    FieldGenerator.h
    GraphicsOutput.h
    Parameters.h
    ParticleGenerator.h
    Random.h
    Timer.h

    Random.cpp
    Timer.cpp)
