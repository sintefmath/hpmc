/* Copyright STIFTELSEN SINTEF 2012
 *
 * Authors: Christopher Dyken <christopher.dyken@sintef.no>
 *
 * This file is part of the HPMC library.
 *
 * The HPMC library is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License ("GPL") as published by the
 * Free Software Foundation, either version 2 of the License, or (at your
 * option) any later version.
 *
 * The HPMC library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * the HPMC library.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef HPMC_APP_COMMON_HPP
#define HPMC_APP_COMMON_HPP

#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <hpmc.h>

// === Prototypes for functions provided by the individual app =================
void
render( float t, float dt, float fps, const GLfloat* P, const GLfloat* MV, const GLfloat* PM, const GLfloat *NM );

void
init( int argc, char** argv );

void
printHelp( const std::string& appname );

const std::string
infoString( float fps );

// === Prototypes for functions provided by common.cpp =========================

extern double               aspect_x;
extern double               aspect_y;
extern bool                 wireframe;
extern bool                 record;
extern bool                 binary;
extern HPMCTarget           hpmc_target;
extern HPMCDebugBehaviour   hpmc_debug;


void
printOptions();

void
frustum( GLfloat* dst, GLfloat l, GLfloat r, GLfloat b, GLfloat t, GLfloat n, GLfloat f );

void
translate( GLfloat* dst, GLfloat x, GLfloat y, GLfloat z );

void
rotX( GLfloat* dst, GLfloat degrees );

void
rotY( GLfloat* dst, GLfloat degrees );

void
extractUpperLeft3x3( GLfloat* dst, GLfloat* src );

void
rightMulAssign( GLfloat* A, GLfloat* B );

void
checkFramebufferStatus( const std::string& file, const int line );

void
compileShader( GLuint shader, const std::string& what );

void
linkProgram( GLuint program, const std::string& what );



#endif // of HPMC_APP_COMMON_HPP
