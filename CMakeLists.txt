# cmake_minimum_required(VERSION 3.9)

# common options

enable_language(CXX)

# set(CMAKE_C_COMPILER   "/usr/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -W -Wall -Wextra -Wpedantic \
      -Waggressive-loop-optimizations \
      -Wcast-align -Wcast-qual \
      -Wdouble-promotion -Wduplicated-branches -Wduplicated-cond \
      -Wfloat-equal -Wformat=2 -Wformat-signedness -Wframe-larger-than=32768 \
      -Wlogical-op \
      -Wnull-dereference \
      -Wodr -Wold-style-cast \
      -Wshadow=local -Wshift-overflow=2 -Wstrict-aliasing=2 -Wsuggest-final-methods -Wsuggest-final-types -Wsync-nand \
      -Wtrampolines \
      -Wuseless-cast -Wno-unused-but-set-parameter \
      -Wwrite-strings"
      CACHE INTERNAL ""
  )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -Weverything \
      -Wno-class-varargs -Wno-padded \
      -Wc++17-compat \
      -Wno-c++98-compat -Wno-c++98-compat-pedantic"
      # -Wno-c++14-extensions
      # -Wno-c++17-extensions
      # -Wno-switch-enum -Wno-unused-macros
      CACHE INTERNAL ""
  )
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g"
    CACHE INTERNAL ""
)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} \
    -g -pg -no-pie \
    -O3 -ffast-math -fno-rtti"
    CACHE INTERNAL ""
)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
    "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -pg"
    CACHE INTERNAL ""
)

# rolled loops works faster here
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
    -O3 -ffast-math -fno-rtti"
    CACHE INTERNAL ""
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os"
      CACHE INTERNAL "")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Oz"
      CACHE INTERNAL "")
endif()

set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} \
    -O3 -ffast-math -fno-math-errno -fno-unroll-loops \
    -fno-rtti -fno-stack-protector -fno-ident \
    -fomit-frame-pointer -ffunction-sections -fdata-sections \
    -fmerge-all-constants"
    CACHE INTERNAL ""
)

# project configuration

project(kerasify)

include_directories(kerasify PRIVATE "./include")

add_executable(kerasify "")
target_sources(kerasify PRIVATE
    src/layer.cpp
    src/layers/activation.cpp
    src/layers/conv1d.cpp
    src/layers/conv2d.cpp
    src/layers/dense.cpp
    src/layers/elu.cpp
    src/layers/embedding.cpp
    src/layers/flatten.cpp
    src/layers/lstm.cpp
    src/layers/locally1d.cpp
    src/layers/locally2d.cpp
    src/layers/maxpooling2d.cpp
    src/layers/normalization.cpp
    src/model.cpp
    src/model_test.cpp
    src/tensor.cpp
    src/utility.cpp
)

# find_package(Threads REQUIRED)
# target_link_libraries(kerasify PRIVATE ${CMAKE_THREAD_LIBS_INIT})
