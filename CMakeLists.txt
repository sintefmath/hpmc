cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT( hpmc )

OPTION( BUILD_TESTS "Build tests that check if platform performs as expected" ON )
OPTION( DEBUG "Create debug build" OFF )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        "${hpmc_SOURCE_DIR}")

INCLUDE_DIRECTORIES(
  "./hpmc/include"
)

FIND_PACKAGE( OpenGL REQUIRED )
FIND_PACKAGE( GLUT REQUIRED )
FIND_PACKAGE( GLEW REQUIRED )

SET( HPMC_SRC
     "hpmc/src/hpiface.cpp"
     "hpmc/src/util.cpp"
     "hpmc/src/tables.cpp"
     "hpmc/src/triface.cpp"
     "hpmc/src/buildpass.cpp"
     "hpmc/src/init.cpp"
     "hpmc/src/constants.cpp"
     "hpmc/src/shadergen.cpp"
     "hpmc/src/shaderbuild.cpp"
     "hpmc/src/texfbosetup.cpp"
)
#SET( CMAKE_CXX_FLAGS "-pedantic -O2 -DDEBUG ${CMAKE_CXX_FLAGS}" )
#SET( CMAKE_CXX_FLAGS "-pedantic ${CMAKE_CXX_FLAGS}" )

IF( DEBUG )
    SET( CMAKE_CXX_FLAGS "-DDEBUG ${CMAKE_CXX_FLAGS}" )
ENDIF( DEBUG )

ADD_LIBRARY( hpmc STATIC ${HPMC_SRC} )
TARGET_LINK_LIBRARIES( hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

INCLUDE_DIRECTORIES( ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} )

IF( BUILD_TESTS )
    ADD_EXECUTABLE( test_simplemipmapbuildup "tests/simplemipmapbuildup.cpp" )
    TARGET_LINK_LIBRARIES( test_simplemipmapbuildup ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
ENDIF( BUILD_TESTS )

ADD_EXECUTABLE( texture3d "apps/texture3d/texture3d.cpp" )
TARGET_LINK_LIBRARIES( texture3d hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( cayley "apps/cayley/cayley.cpp" )
TARGET_LINK_LIBRARIES( cayley hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( transform_feedback "apps/transform_feedback/transform_feedback.cpp" )
TARGET_LINK_LIBRARIES( transform_feedback hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( metaballs "apps/metaballs/metaballs.cpp" )
TARGET_LINK_LIBRARIES( metaballs hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( particles "apps/particles/particles.cpp" )
TARGET_LINK_LIBRARIES( particles hpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
