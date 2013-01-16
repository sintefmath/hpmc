cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT( hpmc )

SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules )

OPTION( SINTEF_INTERNAL "Compiled internally at SINTEF" OFF )
OPTION( ENABLE_DEBUG "Enable debug code" ON )

IF( UNIX )
    SET( CMAKE_CXX_FLAGS "-pedantic -O2 ${CMAKE_CXX_FLAGS}" )
    IF( ENABLE_DEBUG )
        SET( CMAKE_CXX_FLAGS "-DDEBUG ${CMAKE_CXX_FLAGS}" )
    ENDIF( ENABLE_DEBUG )
    IF( SINTEF_INTERNAL )
        SET( CMAKE_CXX_FLAGS "-DSINTEF_INTERNAL ${CMAKE_CXX_FLAGS}" )
    ENDIF( SINTEF_INTERNAL )
ENDIF( UNIX )

IF( WIN32 )
    ADD_DEFINITIONS( -D_USE_MATH_DEFINES )
ENDIF( WIN32 )


FIND_PACKAGE( OpenGL REQUIRED )
FIND_PACKAGE( GLUT REQUIRED )
FIND_PACKAGE( GLEW REQUIRED )
INCLUDE( txt2cpp )

INCLUDE_DIRECTORIES(
  "./hpmc/include"
  "./hpmc/src"
)
SET( HPMC_SRCS
    "hpmc/src/util.cpp"
    "hpmc/src/tables.cpp"
    "hpmc/src/shadergen.cpp"
    "hpmc/src/shaderbuild.cpp"
    "hpmc/src/Constants.hpp"
    "hpmc/src/Constants.cpp"
    "hpmc/src/VertexCountTable.cpp"
    "hpmc/src/GPGPUQuad.cpp"
    "hpmc/src/SequenceRenderer.cpp"
    "hpmc/src/IntersectingEdgeTable.cpp"
    "hpmc/src/IsoSurface.cpp"
    "hpmc/src/IsoSurfaceRenderer.cpp"
    "hpmc/src/HistoPyramid.cpp"
    "hpmc/src/BaseLevelBuilder.cpp"
    "hpmc/src/Field.cpp"
    "hpmc/src/Logger.cpp"
    "hpmc/src/PublicAPI.cpp"
)
SOURCE_GROUP( "HPMC sources" FILES ${HPMC_SRCS} )

SET( HPMC_HDRS
    "hpmc/include/hpmc.h"
    "hpmc/src/hpmc_internal.h"
    "hpmc/src/Constants.hpp"
    "hpmc/src/VertexCountTable.hpp"
    "hpmc/src/GPGPUQuad.hpp"
    "hpmc/src/SequenceRenderer.hpp"
    "hpmc/src/IntersectingEdgeTable.hpp"
    "hpmc/src/IsoSurface.hpp"
    "hpmc/src/IsoSurfaceRenderer.hpp"
    "hpmc/src/HistoPyramid.hpp"
    "hpmc/src/BaseLevelBuilder.hpp"
    "hpmc/src/Field.hpp"
    "hpmc/src/Logger.hpp"
)
SOURCE_GROUP( "HPMC headers" FILES ${HPMC_HDRS} )
SET_SOURCE_FILES_PROPERTIES( ${HPMC_HDRS} PROPERTIES HEADER_FILE_ONLY TRUE )

# HPMC library
ADD_LIBRARY( hpmc STATIC ${HPMC_SRCS} ${HPMC_HDRS} )
TARGET_LINK_LIBRARIES( hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )


# example apps
INCLUDE_DIRECTORIES( ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} )

SET( HPMC_APP_COMMON_SRCS "apps/common/common.cpp" )
SOURCE_GROUP( "Apps common sources" FILES ${HPMC_APP_COMMON_SRCS} )
SET( HPMC_APP_COMMON_HDRS "apps/common/common.hpp" )
SOURCE_GROUP( "Apps common headers" FILES ${HPMC_APP_COMMON_HDRS} )
SET_SOURCE_FILES_PROPERTIES( ${HPMC_APP_COMMON_HDRS} PROPERTIES HEADER_FILE_ONLY TRUE )
SET( HPMC_APP_COMMON_GLSL "apps/shaders/phong_fs_110.glsl"
                          "apps/shaders/phong_fs_130.glsl"
                          "apps/shaders/phong_vs_110.glsl"
                          "apps/shaders/phong_vs_130.glsl"
                          "apps/shaders/solid_fs_110.glsl"
                          "apps/shaders/solid_fs_130.glsl"
                          "apps/shaders/solid_vs_110.glsl"
                          "apps/shaders/solid_vs_130.glsl"
                          "apps/shaders/shiny_fs_110.glsl"
                          "apps/shaders/shiny_fs_130.glsl"
                          "apps/shaders/shiny_vs_110.glsl"
                          "apps/shaders/shiny_vs_130.glsl"
                          "apps/shaders/plain_vs_110.glsl"
                          "apps/shaders/plain_vs_130.glsl"
                          "apps/shaders/metaballs_fetch.glsl"
                          "apps/shaders/cayley_fetch.glsl"
)
SOURCE_GROUP( "Apps common shaders" FILES ${HPMC_APP_COMMON_GLSL} )
SET_SOURCE_FILES_PROPERTIES( ${HPMC_APP_COMMON_GLSL} PROPERTIES HEADER_FILE_ONLY TRUE )
FOREACH( file ${HPMC_APP_COMMON_GLSL} )
    GET_FILENAME_COMPONENT( symbol ${file} NAME_WE)
    ADD_TEXT_FILE( ${symbol} "resources::${symbol}" ${file} )
    LIST( APPEND HPMC_APP_COMMON_GLSL_LIBS ${symbol} )
ENDFOREACH()
ADD_LIBRARY( hpmc_app_common STATIC ${HPMC_APP_COMMON_SRCS} ${HPMC_APP_COMMON_HDRS} ${HPMC_APP_COMMON_GLSL} )
TARGET_LINK_LIBRARIES( hpmc_app_common ${HPMC_APP_COMMON_GLSL_LIBS} )


ADD_EXECUTABLE( texture3d "apps/texture3d/texture3d.cpp" )
TARGET_LINK_LIBRARIES( texture3d hpmc hpmc_app_common ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( cayley "apps/cayley/cayley.cpp" )
TARGET_LINK_LIBRARIES( cayley hpmc hpmc_app_common ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( transform_feedback "apps/transform_feedback/transform_feedback.cpp" ${HPMC_APP_COMMON_SRC} )
TARGET_LINK_LIBRARIES( transform_feedback hpmc_app_common hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( metaballs "apps/metaballs/metaballs.cpp" )
TARGET_LINK_LIBRARIES( metaballs hpmc hpmc_app_common ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

#ADD_EXECUTABLE( splatting "apps/splatting/splatting.cpp" ${HPMC_APP_COMMON_SRC} )
#TARGET_LINK_LIBRARIES( splatting hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

#ADD_EXECUTABLE( particles "apps/particles/particles.cpp" ${HPMC_APP_COMMON_SRC} )
#TARGET_LINK_LIBRARIES( particles hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
