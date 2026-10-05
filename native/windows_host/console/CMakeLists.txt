cmake_minimum_required(VERSION 3.21)
project(puripuly_console LANGUAGES CXX RC)
get_filename_component(PURIPULY_APP_ICON
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/puripuly_heart/data/icons/icon.ico"
    ABSOLUTE
)
configure_file(app_icon.rc.in "${CMAKE_CURRENT_BINARY_DIR}/app_icon.rc" @ONLY)
add_executable(puripuly main.cpp "${CMAKE_CURRENT_BINARY_DIR}/app_icon.rc")
target_compile_features(puripuly PRIVATE cxx_std_17)
target_link_libraries(puripuly PRIVATE shell32)
