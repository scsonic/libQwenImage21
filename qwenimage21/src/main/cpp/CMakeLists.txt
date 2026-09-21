cmake_minimum_required(VERSION 3.18)
project(qwenimage21_jni CXX)

# Prebuilt libMNN.so (scsonic/MNN branch qwen-image-21, built by scripts/build_libmnn_android.sh)
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so)

add_library(qwenimage21_jni SHARED qwenimage21_jni.cpp)
target_include_directories(qwenimage21_jni PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_options(qwenimage21_jni PRIVATE -std=c++17 -fexceptions)
target_link_libraries(qwenimage21_jni MNN log android)
