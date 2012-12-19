/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc_internal.h
 *
 *  Created: 24. June 2009
 *
 *  Version: $Id: $
 *
 *  Authors: Christopher Dyken <christopher.dyken@sintef.no>
 *
 *  This file is part of the HPMC library.
 *  Copyright (C) 2009 by SINTEF.  All rights reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  ("GPL") version 2 as published by the Free Software Foundation.
 *  See the file LICENSE.GPL at the root directory of this source
 *  distribution for additional information about the GNU GPL.
 *
 *  For using HPMC with software that can not be combined with the
 *  GNU GPL, please contact SINTEF for aquiring a commercial license
 *  and support.
 *
 *  SINTEF, Pb 124 Blindern, N-0314 Oslo, Norway
 *  http://www.sintef.no
 *********************************************************************/
/**
  * \file hpmc_internal.h
  *
  * \brief Library internal header file. Defines internal interface.
  *
  *
  */
#ifndef _HPMC_INTERNAL_H_
#define _HPMC_INTERNAL_H_

#include <GL/glew.h>
#include <string>
#include <vector>
#include <iostream>
#include <hpmc.h>
#include "HistoPyramid.hpp"



/** \addtogroup hpmc_public
  * \{
  */

// Logger interface that should be easy to replace with log4cxx

namespace HPMC {
/*
typedef std::string OldLogger;
static inline OldLogger getLogger( const std::string& component ) { return component; }
#define HPMCLOG_OUTPUT(c,a,b) \
    do { \
        std::cerr << c << a; \
        for(size_t __i=(a).length(); __i<20; __i++) { \
            std::cerr << ' '; \
        } \
        std::cerr << "    " << b << std::endl; \
} while(0)

#define HPMCLOG_TRACE(a,b)
#ifdef DEBUG
#define HPMCLOG_DEBUG(a,b) HPMCLOG_OUTPUT("[D]",a,b)
#define HPMCLOG_INFO(a,b) HPMCLOG_OUTPUT("[I]",a,b)
#else
#define HPMCLOG_DEBUG(a,b)
#define HPMCLOG_INFO(a,b)
#endif
#define HPMCLOG_WARN(a,b)  HPMCLOG_OUTPUT("[W]",a,b)
#define HPMCLOG_ERROR(a,b) HPMCLOG_OUTPUT("[E]",a,b)
#define HPMCLOG_FATAL(a,b) HPMCLOG_OUTPUT("[F]",a,b)
*/

}

#define remapCode( code ) (    \
    ((((code)>>0)&0x1)<<0) |   \
    ((((code)>>1)&0x1)<<1) |   \
    ((((code)>>4)&0x1)<<2) |   \
    ((((code)>>5)&0x1)<<3) |   \
    ((((code)>>3)&0x1)<<4) |   \
    ((((code)>>2)&0x1)<<5) |   \
    ((((code)>>7)&0x1)<<6) |   \
    ((((code)>>6)&0x1)<<7) )



// -----------------------------------------------------------------------------

/** \} */
// -----------------------------------------------------------------------------
/** \defgroup hpmc_internal Internal API
  * \{
  */


extern int      HPMC_triangle_table[256][16];

extern GLfloat  HPMC_edge_table[12][4];

extern GLfloat  HPMC_gpgpu_quad_vertices[3*4];

extern GLfloat HPMC_midpoint_table[12][3];



/** Checks field and grid sizes and determine HistoPyramid layout and tiling.
  *
  * \sideeffect None.
 */
bool
HPMCdetermineLayout( struct HPMCIsoSurface* h );



/** Build reduction shaders.
  *
  * \sideeffect GL_CURRENT_PROGRAM
  */
bool
HPMCbuildHPBuildShaders( struct HPMCIsoSurface* h );

/*
bool
HPMCcheckFramebufferStatus( const std::string& file, const int line );

bool
HPMCcheckFramebufferStatus( HPMC::Logger log );
*/
//bool
//HPMCcheckGL( HPMC::OldLogger log );

//bool
//HPMCcheckGL( const std::string& file, const int line );

std::string
HPMCaddLineNumbers( const std::string& src );


GLuint
HPMCcompileShader( const std::string& src, GLuint type );



bool
HPMClinkProgram( GLuint program );

GLint
HPMCgetUniformLocation( GLuint program, const std::string& name );

std::string
HPMCgenerateDefines(const HPMCIsoSurface *h );


std::string
HPMCgenerateExtractVertexFunction( struct HPMCIsoSurface* h );




//void
//HPMCsetLayout( struct HPMCIsoSurface* h );


/** \} */

#endif // _HPMC_INTERNAL_H_
