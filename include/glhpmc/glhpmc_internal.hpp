#pragma once
/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
 *
 * Author(s): Christopher Dyken, <christopher.dyken@sintef.no>
 *
 * HPMC is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * HPMC is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * HPMC.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
  * \file hpmc_internal.h
  *
  * \brief Library internal header file. Defines internal interface.
  *
  *
  */

#include <GL/glew.h>
#include <string>
#include <vector>
#include <iostream>
#include <glhpmc/glhpmc.hpp>
#include "HistoPyramid.hpp"



/** \addtogroup hpmc_public
  * \{
  */

// Logger interface that should be easy to replace with log4cxx

namespace glhpmc {
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
} // of namespace glhpmc
