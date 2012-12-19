cmake_minimum_required(VERSION 2.4)
PROJECT( hpmc )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
	"${hpmc_SOURCE_DIR}")

INCLUDE_DIRECTORIES(
  "./hpmc/include"
)

FIND_PACKAGE( OpenGL REQUIRED )
FIND_PACKAGE( GLUT REQUIRED )
FIND_PACKAGE( GLEW REQUIRED )

SET( HPMC_SRC
     "hpmc/src/hpmc_handle.cpp"
     "hpmc/src/hpmc_util.cpp"
     "hpmc/src/hpmc_shadergen.cpp"
     "hpmc/src/hpmc_tables.cpp"
     "hpmc/src/hpmc_traversal.cpp"
     "hpmc/src/hpmc_build.cpp"
     "hpmc/src/hpmc_init.cpp"
     "hpmc/src/hpmc_constants.cpp"
)
SET( CMAKE_CXX_FLAGS "-DDEBUG ${CMAKE_CXX_FLAGS}" )

ADD_LIBRARY( hpmc_0.1a STATIC ${HPMC_SRC} )
TARGET_LINK_LIBRARIES( hpmc_0.1a ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

INCLUDE_DIRECTORIES( ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} )

ADD_EXECUTABLE( texture3D "apps/texture3D/texture3D.cpp" )
TARGET_LINK_LIBRARIES( texture3D hpmc_0.1a ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

ADD_EXECUTABLE( formula "apps/formula/formula.cpp" )
TARGET_LINK_LIBRARIES( formula hpmc_0.1a ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
