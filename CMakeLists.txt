cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT( hpmc )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        "${hpmc_SOURCE_DIR}")

OPTION( SINTEF_INTERNAL "Compiled internally at SINTEF" OFF )
OPTION( ENABLE_DEBUG "Enable debug code" ON )

INCLUDE_DIRECTORIES(
  "./hpmc/include"
  "./hpmc/src"
)

FIND_PACKAGE( OpenGL REQUIRED )
FIND_PACKAGE( GLUT REQUIRED )
FIND_PACKAGE( GLEW REQUIRED )

SET( HPMC_SRC
     "hpmc/src/util.cpp"
     "hpmc/src/tables.cpp"
     "hpmc/src/shadergen.cpp"
     "hpmc/src/shaderbuild.cpp"
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

IF( UNIX )
    SET( CMAKE_CXX_FLAGS "-pedantic -O2 ${CMAKE_CXX_FLAGS}" )
    IF( ENABLE_DEBUG )
        SET( CMAKE_CXX_FLAGS "-DDEBUG ${CMAKE_CXX_FLAGS}" )
    ENDIF( ENABLE_DEBUG )
    IF( SINTEF_INTERNAL )
        SET( CMAKE_CXX_FLAGS "-DSINTEF_INTERNAL ${CMAKE_CXX_FLAGS}" )
    ENDIF( SINTEF_INTERNAL )
ENDIF( UNIX )

ADD_LIBRARY( hpmc STATIC ${HPMC_SRC} )
TARGET_LINK_LIBRARIES( hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

INCLUDE_DIRECTORIES( ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} )

ADD_EXECUTABLE( texture3d "apps/texture3d/texture3d.cpp" )
TARGET_LINK_LIBRARIES( texture3d hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( cayley "apps/cayley/cayley.cpp" )
TARGET_LINK_LIBRARIES( cayley hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( transform_feedback "apps/transform_feedback/transform_feedback.cpp" )
TARGET_LINK_LIBRARIES( transform_feedback hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( metaballs "apps/metaballs/metaballs.cpp" )
TARGET_LINK_LIBRARIES( metaballs hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( splatting "apps/splatting/splatting.cpp" )
TARGET_LINK_LIBRARIES( splatting hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( particles "apps/particles/particles.cpp" )
TARGET_LINK_LIBRARIES( particles hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
