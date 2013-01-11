/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: util.cpp
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

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <hpmc.h>
#include <hpmc_internal.h>

using std::string;
using std::vector;
using std::stringstream;
using std::setw;
using std::cerr;
using std::endl;

// -----------------------------------------------------------------------------
#define HELPER(a) case a: error = #a; break
bool
HPMCcheckFramebufferStatus( HPMCConstants* c, const std::string& file, const int line )
{
    if( c->m_target < HPMC_TARGET_GL30_GLSL130 ) {
        GLenum status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
        if( status == GL_FRAMEBUFFER_COMPLETE_EXT ) {
            return true;
        }

#ifdef DEBUG
        string error = "unknown error";
        switch( status ) {
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT );
        HELPER( GL_FRAMEBUFFER_UNSUPPORTED_EXT );
        }
        cerr << "HPMC error: framebuffer incomplete (" << error << ") in "
             << file << " at line " << line << "." << endl;
#endif
    }
    else {
        GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
        if( status == GL_FRAMEBUFFER_COMPLETE ) {
            return true;
        }
#ifdef DEBUG
        string error = "unknown error";
        switch( status ) {
        HELPER( GL_FRAMEBUFFER_UNDEFINED );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER );
        HELPER( GL_FRAMEBUFFER_UNSUPPORTED );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE );
        HELPER( GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS );
        }
        cerr << "HPMC error: framebuffer incomplete (" << error << ") in "
             << file << " at line " << line << "." << endl;
#endif
    }
    return false;
}

// -----------------------------------------------------------------------------
bool
HPMCcheckGL( const std::string& file, const int line )
{
    GLenum glerror = glGetError();
    if( glerror == GL_NO_ERROR ) {
        return true;
    }

#ifdef DEBUG
    string error = "unknown error";
    switch( glerror ) {
        HELPER( GL_INVALID_ENUM );
        HELPER( GL_INVALID_VALUE );
        HELPER( GL_INVALID_OPERATION );
        HELPER( GL_STACK_OVERFLOW );
        HELPER( GL_STACK_UNDERFLOW );
        HELPER( GL_OUT_OF_MEMORY );
        HELPER( GL_TABLE_TOO_LARGE );
        HELPER( GL_INVALID_FRAMEBUFFER_OPERATION );
    }
    cerr << "HPMC warning: error on gl state ("
         << error << ", code="<<glerror<<") in "
         << file << " at line " << line << "." << endl;
#endif

    return false;
}
#undef HELPER

// -----------------------------------------------------------------------------
std::string
HPMCaddLineNumbers( const std::string& src )
{
    stringstream out;

    int line = 1;
    for( string::const_iterator it = src.begin(); it!=src.end(); ++it ) {
        string::const_iterator jt = it;
        int c=0;
        out << std::setw(3) << line << ": ";
        for(; *jt != '\n' && jt != src.end(); jt++) {
            out << *jt;
            c++;
        }
        out << "\n";
        line ++;
        it = jt;
        if(jt == src.end() )
            break;
    }

    return out.str();
}

// -----------------------------------------------------------------------------
GLuint
HPMCcompileShader( const std::string& src, GLuint type )
{
    GLuint shader = glCreateShader( type );

    // glShaderSource wants an array of string pointers
    const char* p = src.c_str();
    glShaderSource( shader, 1, &p, NULL );
    glCompileShader( shader );

    // check if everything is ok
    GLint status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
    if( status == GL_TRUE ) {
        // successful compilation
        return shader;
    }

    // compilation failed
#ifdef DEBUG
    cerr << "HPMC error: compilation of shader failed." << endl;
    cerr << "HPMC error: *** shader source code ***" << endl;
    cerr << HPMCaddLineNumbers( src );
    cerr << "HPMC error: *** shader build log ***" << endl;

    // get size of build log
    GLint logsize;
    glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );

    // get build log
    if( logsize == 0 ) {
        cerr << "HPMC error: empty log." << endl;
    }
    else {
        vector<GLchar> infolog( logsize+1 );
        glGetShaderInfoLog( shader, logsize, NULL, &infolog[0] );
        cerr << string( infolog.begin(), infolog.end() ) << endl;
    }
#endif

    glDeleteShader( shader );
    return 0u;
}

// -----------------------------------------------------------------------------
bool
HPMClinkProgram( GLuint program )
{
    glLinkProgram( program );

    GLint linkstatus;
    glGetProgramiv( program, GL_LINK_STATUS, &linkstatus );
    if( linkstatus == GL_TRUE ) {
        // successful link
        return true;
    }

    // linking failed
#ifdef DEBUG
    cerr << "HPMC: linking of program failed." << endl;
    cerr << "HPMC: *** program link log ***" << endl;

    // get size of build log
    GLint logsize;
    glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );

    // get build log
    if( logsize == 0 ) {
        cerr << "HPMC: empty log." << endl;
    }
    else {
        vector<GLchar> infolog( logsize+1 );
        glGetProgramInfoLog( program, logsize, NULL, &infolog[0] );
        cerr << string( infolog.begin(), infolog.end() ) << endl;
    }
#endif

    return false;
}

// -----------------------------------------------------------------------------
GLint
HPMCgetUniformLocation( GLuint program, const std::string& name )
{
    GLint loc = glGetUniformLocation( program, name.c_str() );
#ifdef DEBUG
    if( loc < 0 ) {
        cerr << "HPMC warning: failed to locate uniform \"" << name << "\"." << endl;
    }
#endif
    return loc;
}

// -----------------------------------------------------------------------------
void
HPMCrenderGPGPUQuad( struct HPMCHistoPyramid* h )
{
    glBindBuffer( GL_ARRAY_BUFFER, h->m_constants->m_gpgpu_quad_vbo );
    glVertexPointer( 3, GL_FLOAT, 0, NULL );
    glEnableClientState( GL_VERTEX_ARRAY );
    glDrawArrays( GL_QUADS, 0, 4 );
}
