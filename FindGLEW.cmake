#Tries to find GLEW library
#
# Copyright <André Rigland Brodtkorb> Andre.Brodtkorb@sintef.no
#

#Find glew library
FIND_LIBRARY(GLEW_LIBRARIES 
	NAMES GLEW glew glew32
	PATHS "/usr/lib"
	"/usr/lib64"
	"$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/Lib"
	"$ENV{ProgramFiles}/Microsoft Visual Studio 9.0/VC/lib/"
)

#Find glew header
FIND_PATH(GLEW_INCLUDE_DIR "GL/glew.h"
	"/usr/include"
	"$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/Include"
	"$ENV{ProgramFiles}/Microsoft Visual Studio 9.0/VC/include/"
)

#check that we have found everything
SET(GLEW_FOUND "NO")
IF(GLEW_LIBRARIES)
  IF(GLEW_INCLUDE_DIR)
    SET(GLEW_FOUND "YES")
  ENDIF(GLEW_INCLUDE_DIR)
ENDIF(GLEW_LIBRARIES)

#Mark options as advanced
MARK_AS_ADVANCED(
  GLEW_INCLUDE_DIR
  GLEW_LIBRARIES
)

#SET (GLEW_LIBRARY ON CACHE INTERNAL "yo")

IF (GLEW_LIBRARIES)
	SET (GLEW_LIBRARY ${GLEW_LIBRARIES} CACHE INTERNAL "Deprecated variable shadows GLEW_LIBRARIES")
ENDIF (GLEW_LIBRARIES)