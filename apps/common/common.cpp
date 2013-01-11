/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: common.cpp
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

#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>
#include <GL/glew.h>
#if defined(__unix) || defined(__APPLE__)
#include <sys/time.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>
#define snprintf _snprintf_s
#endif
using std::min;
using std::max;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::copy;
using std::back_insert_iterator;

double aspect_x=1.0;
double aspect_y=1.0;
bool wireframe = false;

// -----------------------------------------------------------------------------
#define ASSERT_GL do {                                                         \
    GLenum err = glGetError();                                                 \
    if( err != GL_NO_ERROR ) {                                                 \
        cerr << __FILE__ << '@' << __LINE__ << ": OpenGL error:"               \
             << err << endl;                                                   \
        exit( EXIT_FAILURE );                                                  \
    }                                                                          \
} while(0);

// -----------------------------------------------------------------------------
double
getTimeOfDay()
{
#if defined(__unix) || defined(__APPLE__)
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec+tv.tv_usec*1e-6;
#elif defined(_WIN32)
    LARGE_INTEGER f;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return t.QuadPart/(double) f.QuadPart;
#else
    return 0;
#endif
}

// --- compile shader and check for errors -------------------------------------
void
compileShader( GLuint shader, const string& what )
{
    glCompileShader( shader );

    GLint compile_status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &compile_status );
    if( compile_status != GL_TRUE ) {
        cerr << "Compilation of " << what << " failed, infolog:" << endl;

        GLint logsize;
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            vector<GLchar> infolog( logsize+1 );
            glGetShaderInfoLog( shader, logsize, NULL, &infolog[0] );
            cerr << string( infolog.begin(), infolog.end() ) << endl;
        }
        else {
            cerr << "Empty log message" << endl;
        }
        cerr << "Exiting." << endl;
        exit( EXIT_FAILURE );
    }
}

// --- compile program and check for errors ------------------------------------
void
linkProgram( GLuint program, const string& what )
{
    glLinkProgram( program );

    GLint linkstatus;
    glGetProgramiv( program, GL_LINK_STATUS, &linkstatus );
    if( linkstatus != GL_TRUE ) {
        cerr << "Linking of " << what << " failed, infolog:" << endl;

        GLint logsize;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            vector<GLchar> infolog( logsize+1 );
            glGetProgramInfoLog( program, logsize, NULL, &infolog[0] );
            cerr << string( infolog.begin(), infolog.end() ) << endl;
        }
        else {
            cerr << "Empty log message" << endl;
        }
        cerr << "Exiting." << endl;
        exit( EXIT_FAILURE );
    }
}

// --- set a list of varyings as active ----------------------------------------
void
activateVaryings( GLuint program,
                  GLsizei count,
                  const GLchar** names )
{
    for(GLsizei i=0; i<count; i++) {
        glActiveVaryingNV( program, names[i] );
    }
}

// --- enable a list of varyings for feedback transform ------------------------
void
setFeedbackVaryings( GLuint program,
                     GLsizei count,
                     const GLchar** names )
{
    vector<GLint> locs( count );
    for( GLsizei i=0; i<count; i++ ) {
        locs[i] = glGetVaryingLocationNV( program, names[i] );
        if( locs[i] == -1 ) {
            cerr << "Failed to get varying location of "
                 << names[i] << "\n";
            exit( EXIT_FAILURE );
        }
    }
    glTransformFeedbackVaryingsNV( program,
                                   count,
                                   &locs[0],
                                   GL_INTERLEAVED_ATTRIBS_NV );
}

// -----------------------------------------------------------------------------
void
keyboard( unsigned char key, int x, int y )
{
    if(key == 'w') {
        wireframe = !wireframe;
    }
    else if(key == 'q') {
        exit( EXIT_SUCCESS );
    }
}

// -----------------------------------------------------------------------------
void
reshape(int w, int h)
{
    if( w > h ) {
        aspect_x = (double)w/(double)h;
        aspect_y = 1.0;
    } else {
        aspect_x = 1.0;
        aspect_y = (double)h/(double)w;
    }
    if( w > 0 && h > 0 ) {
        glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    }
    glutPostRedisplay();
}

// -----------------------------------------------------------------------------
void
idle()
{
        glutPostRedisplay();
}

// --- prototype; the applications implement their own render loop -------------
void
render( float t, float dt, float fps );

// --- calculate fps and call render loop func ---------------------------------
void
display()
{
    ASSERT_GL;

    double t = getTimeOfDay();;
    static double pt;
    static double start;
    static double last_fps_t;
    static int frames = 0;
    static double fps = 0.0;
    static bool first = true;
    if( first ) {
        first = false;
        start = t;
        last_fps_t = pt = t-start;
    }
    t = t-start;
    float dt = max(0.0, min(1.0, t-pt));
    frames++;
    if( t-last_fps_t > (1.0/30.0 ) ) {
        fps = frames / (t-last_fps_t);
        last_fps_t = t;
        frames = 0;
    }
    render( t, dt, fps );
    pt = t;
    glutSwapBuffers();
}

