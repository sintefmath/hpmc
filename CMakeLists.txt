cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT( hpmc )

SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules )

OPTION( ENABLE_GLHPMC "Enable building of OpenGL HPMC" ON )
OPTION( ENABLE_CUHPMC "Enable building of CUDA HPMC" ON )
OPTION( ENABLE_CUHPMC_INTEROP "Build CUDA HPMC with GL interop support" ON )
OPTION( ENABLE_SAMPLES "Enable building of sample applications" ON )

OPTION( SINTEF_INTERNAL "Compiled internally at SINTEF" OFF )
OPTION( ENABLE_DEBUG "Enable debug code" ON )

INCLUDE( txt2cpp )
FIND_PACKAGE( OpenGL REQUIRED )
FIND_PACKAGE( GLUT REQUIRED )
FIND_PACKAGE( GLEW REQUIRED )
IF( ENABLE_CUHPMC )
    OPTION( HPMC_CUDA_GEN_SM20 "Generate compute capability 2.0 code" ON )
    OPTION( HPMC_CUDA_GEN_SM30 "Generate compute capability 3.0 code" ON )
    OPTION( HPMC_CUDA_GEN_FWD  "Generate beyond 3.0 compatible ptx code" OFF )
    OPTION( HPMC_CUDA_DEBUG_GDB "Add flags thad cuda-gdb wants" OFF )
    OPTION( HPMC_CUDA_VERBOSE_PTXAS "Let ptxas be verbose about resource usage" ON )
    FIND_PACKAGE( CUDA REQUIRED )
    SET( HPMC_NVCC_OPTIONS )

    IF( HPMC_CUDA_VERBOSE_PTXAS )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v"  )
    ENDIF( HPMC_CUDA_VERBOSE_PTXAS )

    IF( HPMC_CUDA_DEBUG_GDB )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -lineinfo " )
    ENDIF( HPMC_CUDA_DEBUG_GDB )

    IF( HPMC_CUDA_GEN_SM20 )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_20,code=sm_20" )
    ENDIF( HPMC_CUDA_GEN_SM20 )

    IF( HPMC_CUDA_GEN_SM30 )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_30,code=sm_30" )
    ENDIF( HPMC_CUDA_GEN_SM30 )

    IF( HPMC_CUDA_GEN_FWD )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_30,code=compute_30" )
    ENDIF( HPMC_CUDA_GEN_FWD )

ENDIF( ENABLE_CUHPMC )


IF( UNIX )
    SET( CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}" )
    IF( ENABLE_DEBUG )
        SET( CMAKE_CXX_FLAGS "-DDEBUG -g1 ${CMAKE_CXX_FLAGS}" )
    ENDIF( ENABLE_DEBUG )
    IF( SINTEF_INTERNAL )
        SET( CMAKE_CXX_FLAGS "-DSINTEF_INTERNAL ${CMAKE_CXX_FLAGS}" )
    ENDIF( SINTEF_INTERNAL )
ENDIF( UNIX )
IF( WIN32 )
    ADD_DEFINITIONS( -D_USE_MATH_DEFINES )
ENDIF( WIN32 )


INCLUDE_DIRECTORIES( "./include" )

# --- GLHPMC library -----------------------------------------------------------
IF( ENABLE_GLHPMC )
    FILE( GLOB GLHPMC_HDRS "include/glhpmc/*.hpp" )
    SOURCE_GROUP( "GLHPMC headers" FILES ${GLHPMC_HDRS} )
    SET_SOURCE_FILES_PROPERTIES( ${GLHPMC_HDRS} PROPERTIES HEADER_FILE_ONLY TRUE )

    FILE( GLOB GLHPMC_SRCS "src/glhpmc/*.cpp" )
    SOURCE_GROUP( "GLHPMC sources" FILES ${GLHPMC_SRCS} )

    FILE( GLOB GLHPMC_GLSL "src/glhpmc/shaders/*.glsl" )
    SOURCE_GROUP( "GLHPMC shaders" FILES ${GLHPMC_GLSL} )
    SET_SOURCE_FILES_PROPERTIES( ${GLHPMC_GLSL} PROPERTIES HEADER_FILE_ONLY TRUE )
    FOREACH( file ${GLHPMC_GLSL} )
        GET_FILENAME_COMPONENT( symbol ${file} NAME_WE)
        ADD_RESOURCE_TXT( GLHPMC_GLSL_GEN "glhpmc::resources::${symbol}" ${file} )
    ENDFOREACH()

    ADD_LIBRARY( glhpmc STATIC ${GLHPMC_SRCS} ${GLHPMC_HDRS} ${GLHPMC_GLSL_GEN} ${GLHPMC_GLSL} )
    TARGET_LINK_LIBRARIES( glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
ENDIF( ENABLE_GLHPMC )

# --- CUHPMC library -----------------------------------------------------------
IF( ENABLE_CUHPMC )
    IF( ENABLE_CUHPMC_INTEROP )
        ADD_DEFINITIONS( -DENABLE_CUHPMC_INTEROP )
    ENDIF( ENABLE_CUHPMC_INTEROP )

    INCLUDE_DIRECTORIES( "${CUDA_TOOLKIT_ROOT_DIR}/include" )

    FILE( GLOB CUHPMC_HDRS1 "include/cuhpmc/*.hpp" )
    FILE( GLOB CUHPMC_HDRS2 "src/cuhpmc/kernels/*.hpp" )
    SET( CUHPMC_HDRS ${CUHPMC_HDRS1} ${CUHPMC_HDRS2} )
    SOURCE_GROUP( "CUHPMC headers" FILES ${CUHPMC_HDRS} )

    FILE( GLOB CUHPMC_KRLS "src/cuhpmc/kernels/*.cu" )
    SOURCE_GROUP( "CUHPMC kernels" FILES ${CUHPMC_KRLS} )

    FILE( GLOB CUHPMC_GLSL "src/cuhpmc/shaders/*.glsl" )
    SOURCE_GROUP( "CUHPMC shaders" FILES ${CUHPMC_GLSL} )
    SET_SOURCE_FILES_PROPERTIES( ${CUHPMC_GLSL} PROPERTIES HEADER_FILE_ONLY TRUE )
    FOREACH( file ${CUHPMC_GLSL} )
        GET_FILENAME_COMPONENT( symbol ${file} NAME_WE)
        ADD_RESOURCE_TXT( CUHPMC_GLSL_GEN "cuhpmc::resources::${symbol}" ${file} )
    ENDFOREACH()

    FILE( GLOB CUHPMC_SRCS_CPP "src/cuhpmc/*.cpp" )
    FILE( GLOB CUHPMC_SRCS_CU  "src/cuhpmc/*.cu" )
    SET( CUHPMC_SRCS ${CUHPMC_SRCS_CPP} ${CUHPMC_SRCS_CU} )
    SOURCE_GROUP( "CUHPMC sources" FILES ${CUHPMC_SRCS} )

    CUDA_ADD_LIBRARY( cuhpmc STATIC ${CUHPMC_KRLS} ${CUHPMC_HDRS} ${CUHPMC_SRCS} ${CUHPMC_GLSL_GEN} ${CUHPMC_GLSL} )
    TARGET_LINK_LIBRARIES( cuhpmc )
ENDIF( ENABLE_CUHPMC )

# --- Sample applications ------------------------------------------------------
IF( ENABLE_SAMPLES )

    INCLUDE_DIRECTORIES( ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} )
    SET( HPMC_APP_COMMON_SRCS "apps/common/common.cpp" )
    SOURCE_GROUP( "Apps common sources" FILES ${HPMC_APP_COMMON_SRCS} )

    SET( HPMC_APP_COMMON_HDRS "apps/common/common.hpp" )
    SOURCE_GROUP( "Apps common headers" FILES ${HPMC_APP_COMMON_HDRS} )
    SET_SOURCE_FILES_PROPERTIES( ${HPMC_APP_COMMON_HDRS} PROPERTIES HEADER_FILE_ONLY TRUE )

    FILE( GLOB HPMC_APP_COMMON_GLSL "apps/shaders/*.glsl" )
    SOURCE_GROUP( "Apps common shaders" FILES ${HPMC_APP_COMMON_GLSL} )
    FOREACH( file ${HPMC_APP_COMMON_GLSL} )
        GET_FILENAME_COMPONENT( symbol ${file} NAME_WE)
        ADD_RESOURCE_TXT( HPMC_APP_COMMON_GLSL_GEN "resources::${symbol}" ${file} )
    ENDFOREACH()
    ADD_LIBRARY( hpmc_app_common STATIC ${HPMC_APP_COMMON_SRCS} ${HPMC_APP_COMMON_HDRS} ${HPMC_APP_COMMON_GLSL_GEN} ${HPMC_APP_COMMON_GLSL} )
    TARGET_LINK_LIBRARIES( hpmc_app_common ${HPMC_APP_COMMON_GLSL_LIBS} )

    # --- GLHPMC samples -------------------------------------------------------
    IF( ENABLE_GLHPMC )

        ADD_EXECUTABLE( cayley "apps/cayley/cayley.cpp" )
        TARGET_LINK_LIBRARIES( cayley hpmc_app_common glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

        ADD_EXECUTABLE( texture3d "apps/texture3d/texture3d.cpp" )
        TARGET_LINK_LIBRARIES( texture3d hpmc_app_common glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

        ADD_EXECUTABLE( transform_feedback "apps/transform_feedback/transform_feedback.cpp" ${GLHPMC_APP_COMMON_SRC} )
        TARGET_LINK_LIBRARIES( transform_feedback hpmc_app_common glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

        ADD_EXECUTABLE( metaballs "apps/metaballs/metaballs.cpp" )
        TARGET_LINK_LIBRARIES( metaballs hpmc_app_common glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

        ADD_EXECUTABLE( particles "apps/particles/particles.cpp" ${GLHPMC_APP_COMMON_SRC} )
        TARGET_LINK_LIBRARIES( particles hpmc_app_common glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )

        ADD_EXECUTABLE( splatting "apps/splatting/splatting.cpp" )
        TARGET_LINK_LIBRARIES( splatting hpmc_app_common glhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
    ENDIF( ENABLE_GLHPMC )

    # --- CUHPMC samples -------------------------------------------------------
    IF( ENABLE_CUHPMC )
        CUDA_ADD_EXECUTABLE( texture_cuda "apps/texture_cuda/texture_cuda.cu" )
        TARGET_LINK_LIBRARIES( texture_cuda hpmc_app_common cuhpmc ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
    ENDIF( ENABLE_CUHPMC )

ENDIF( ENABLE_SAMPLES )
