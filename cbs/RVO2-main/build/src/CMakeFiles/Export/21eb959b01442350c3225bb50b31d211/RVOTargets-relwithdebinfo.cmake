#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "RVO::RVO" for configuration "RelWithDebInfo"
set_property(TARGET RVO::RVO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(RVO::RVO PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/RVO.lib"
  )

list(APPEND _cmake_import_check_targets RVO::RVO )
list(APPEND _cmake_import_check_files_for_RVO::RVO "${_IMPORT_PREFIX}/lib/RVO.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
