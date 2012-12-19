/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: texture3D.cpp
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
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GL/glut.h>
#include "hpmc.h"
#include "hpmc_internal.h"

#ifdef __unix
#include <sys/time.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>

#define snprintf _snprintf_s
#endif

double getTimeOfDay();

double aspect_x=1.0;
double aspect_y=1.0;
bool wireframe = false;

int volume_size_x;
int volume_size_y;
int volume_size_z;

GLuint volume_tex;
GLuint onscreen_v;
GLuint onscreen_p;

struct HPMCConstants* hpmc_s;
struct HPMCHistoPyramid* hpmc_h;
struct HPMCTraversalHandle* hpmc_th;

#define ASSERT_GL do {                                                         \
    GLenum err = glGetError();                                                 \
    if( err != GL_NO_ERROR ) {                                                 \
        fprintf( stderr, "%s@%d: GL failed: 0x%x.\n", __FILE__, __LINE__, err ); \
        exit( EXIT_FAILURE );                                                  \
    }                                                                          \
} while(0);

void
display()
{
    ASSERT_GL;

    double t = getTimeOfDay();;
    static double start;
    static bool first = true;
    static int frames;
    static double prevcalc;
    static char message[256];
    frames++;

    if( first ) {
        first = false;
        start = t;
        prevcalc = 0.0;
        frames = 0;
        message[0] = '\0';
    }
    t = t-start;

    // analyze volume
    HPMCbuildHistopyramidUsingTexture( hpmc_h, volume_tex, 0.5 + 0*sin(0.1*t) );
    GLuint N = HPMCacquireNumberOfVertices( hpmc_h );

    if( 1 < t-prevcalc ) {
        double delta = t-prevcalc;
        snprintf( message, 256, "FPS: %d, MVPS: %d, tris: %d",
                  (int)(frames/delta),
                  (int)( (frames*(volume_size_x-0)*(volume_size_y-0)*(volume_size_z-0))/(delta*1e6) ),
                  N/3 );
        frames = 0;
        prevcalc = t;
    }

    // render
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glEnable( GL_DEPTH_TEST );
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable( GL_POLYGON_OFFSET_FILL );
    glEnable( GL_NORMALIZE );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -0.3*aspect_x, 0.3*aspect_x, -0.3*aspect_y, 0.3*aspect_y, 0.5, 6.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0.0f, 0.0f, -3.0f );
    glRotatef( 20, 1.0, 0.0, 0.0 );
    glRotatef( 20.0*t, 0.0, 1.0, 0.0 );
//    glTranslatef( -0.5f, -0.5f, -0.5f );

    // render bounding box

    glPushMatrix();
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            for(int k=0; k<2; k++) {
                glPopMatrix();
                glPushMatrix();
                glTranslatef( i-1.0, j-1.0, k-1.0 );
                glUseProgram( 0 );
                glColor3f( 1.0, 1.0, 1.0 );
                glBegin( GL_LINES );
                glVertex3f( 0.0, 0.0, 0.0 );
                glVertex3f( 1.0, 0.0, 0.0 );
                glVertex3f( 1.0, 0.0, 0.0 );
                glVertex3f( 1.0, 1.0, 0.0 );
                glVertex3f( 1.0, 1.0, 0.0 );
                glVertex3f( 0.0, 1.0, 0.0 );
                glVertex3f( 0.0, 1.0, 0.0 );
                glVertex3f( 0.0, 0.0, 0.0 );
                glVertex3f( 0.0, 0.0, 1.0 );
                glVertex3f( 1.0, 0.0, 1.0 );
                glVertex3f( 1.0, 0.0, 1.0 );
                glVertex3f( 1.0, 1.0, 1.0 );
                glVertex3f( 1.0, 1.0, 1.0 );
                glVertex3f( 0.0, 1.0, 1.0 );
                glVertex3f( 0.0, 1.0, 1.0 );
                glVertex3f( 0.0, 0.0, 1.0 );
                glVertex3f( 1.0, 0.0, 0.0 );
                glVertex3f( 1.0, 0.0, 1.0 );
                glVertex3f( 1.0, 1.0, 0.0 );
                glVertex3f( 1.0, 1.0, 1.0 );
                glVertex3f( 0.0, 1.0, 0.0 );
                glVertex3f( 0.0, 1.0, 1.0 );
                glVertex3f( 0.0, 0.0, 0.0 );
                glVertex3f( 0.0, 0.0, 1.0 );
                glEnd();

                // render solid surface
                glColor3f( 0.5, 0.5, 0.5 );

               glUseProgram( onscreen_p );
               HPMCextractVertices( hpmc_th, N );
           }
        }
    }
    glPopMatrix();


    // render wireframe
    glDisable( GL_POLYGON_OFFSET_FILL );
    if( wireframe ) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glColor3f( 1.0, 1.0, 1.0 );
        HPMCextractVertices( hpmc_th, N );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // render text
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glDisable( GL_DEPTH_TEST );
    glColor3f( 1.0, 1.0, 1.0 );
    glRasterPos2f( -0.9, 0.9 );
    for(int i=0; i<255 && message[i] != '\0'; i++) {
        glutBitmapCharacter( GLUT_BITMAP_HELVETICA_10, (int)message[i] );
    }

    glutSwapBuffers();
}


// a small vertex shader that calls the provided extraction function
std::string custom_vertex_shader =
        "void\n"                                                               \
        "main()\n"                                                             \
        "{\n"                                                                  \
        "    vec3 p, n;\n"                                                     \
        "    extractVertex( p, n );\n"                                         \
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"   \
        "    vec3 cn = normalize( gl_NormalMatrix * n );\n"                    \
        "    float diff = max( 0.4, dot( cn, vec3(0.0, 0.0, 1.0) ) );\n"    \
        "    float spec = pow( max( 0.0, dot( vec3(0.0, 0.0, 1.0), 0.5*(vec3(0.0, 0.0, 1.0)+cn) ) ), 20.0);\n"      \
        "    gl_FrontColor = diff * gl_Color + vec4(spec);\n"                              \
        "}\n";

void
init()
{
    // build volume
    std::vector<GLfloat> buffer( 4*volume_size_x*volume_size_y*volume_size_z );

    for(int k=0; k<volume_size_z; k++) {
        for(int j=0; j<volume_size_y; j++) {
            for(int i=0; i<volume_size_x; i++) {
                bool in = false;

                 if( i < 1  && j < 1 && k < 1 ) {
                    in = true;
                }

                 if( i < 1  && j < 1 && k > volume_size_z-2  ) {
                    in = true;
                }

                 if( i < 1  && j > volume_size_y-2 && k < 1 ) {
                    in = true;
                }

                 if( i < 1  && j > volume_size_y-2 && k > volume_size_z-2 ) {
                    in = true;
                }

                 if(  i > volume_size_x-2&& j < 1 && k < 1 ) {
                     in = true;
                 }

                 if(  i > volume_size_x-2  && j < 1 && k > volume_size_z-2  ) {
                     in = true;
                 }

                 if( i > volume_size_x-2  && j > volume_size_y-2 && k < 1 ) {
                     in = true;
                 }

                 if(  i > volume_size_x-2  && j > volume_size_y-2 && k > volume_size_z-2 ) {
                     in = true;
                 }


                 if( i > volume_size_x-2 && j > volume_size_y-2 && k > volume_size_z-2 ) {
                     in = true;
                 }

                if( (abs(i-volume_size_x/2) < volume_size_x/8) &&
                    (abs(j-volume_size_y/2) < volume_size_y/8)  ) {
                    in = true;
                }
                if( (abs(i-volume_size_x/2) < volume_size_x/8) &&
                    (abs(k-volume_size_z/2) < volume_size_z/8)  ) {
                    in = true;
                }
                if( (abs(j-volume_size_y/2) < volume_size_y/8) &&
                    (abs(k-volume_size_z/2) < volume_size_z/8)  ) {
                    in = true;
                }
                buffer[ (k*volume_size_y + j)*volume_size_x + i ] = 10*in;
            }
        }
    }

    // build volume texture
    glGenTextures( 1, &volume_tex );
    glBindTexture( GL_TEXTURE_3D, volume_tex );
    glTexImage3D( GL_TEXTURE_3D, 0, GL_ALPHA32F_ARB,
                  volume_size_x, volume_size_y, volume_size_z, 0,
                  GL_ALPHA, GL_FLOAT, &buffer[0] );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_3D, 0 );

    hpmc_s = HPMCcreateSingleton();

    hpmc_h = HPMCcreateHistoPyramid2( hpmc_s,
                                      HPMC_VOLUME_LAYOUT_TEXTURE_3D,
                                      HPMC_TAG_WIDTH,  volume_size_x,
                                      HPMC_TAG_HEIGHT, volume_size_y,
                                      HPMC_TAG_DEPTH,  volume_size_z,
                                      HPMC_TAG_END );


    // create basic HP structure
    hpmc_h = HPMCcreateHistoPyramid( hpmc_s,
//                                     HPMC_VOLUME_LAYOUT_FUNCTION,
                                     HPMC_VOLUME_LAYOUT_TEXTURE_3D,
                                     GL_FLOAT,
                                     volume_size_x,
                                     volume_size_y,
                                     volume_size_z );

    // build traversal shader
    hpmc_th = HPMCcreateTraversalHandle( hpmc_h );

    // get a string with HP traversal++-functions
    char* funcs = HPMCgetTraversalShaderFunctions( hpmc_th );

    // build shader
    onscreen_v = HPMCcompileShader( std::string( funcs ) +
                                           custom_vertex_shader,
                                           GL_VERTEX_SHADER );
    free( funcs );

    // link (using vanilla OpenGL vertex shader)
    onscreen_p = glCreateProgram();
    glAttachShader( onscreen_p, onscreen_v );
    glLinkProgram( onscreen_p );

    // check link status
    GLint linkstatus;
    glGetProgramiv( onscreen_p, GL_LINK_STATUS, &linkstatus );
    if( linkstatus != GL_TRUE ) {
        std::cerr << "OpenGL Link error:" << std::endl;

        GLint logsize;
        glGetProgramiv( onscreen_p, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            std::vector<GLchar> infolog( logsize+1 );
            glGetProgramInfoLog( onscreen_p, logsize, NULL, &infolog[0] );
            std::cerr << std::string( infolog.begin(), infolog.end() ) << std::endl;
        }
        else {
            std::cerr << "Empty log." << std::endl;
        }
    }

    HPMClinkProgram( onscreen_p );

    // let HPMC initialize it's uniforms
    HPMCinitializeTraversalHandle( hpmc_th, onscreen_p, 0, 1, 2 );

    glPolygonOffset( 1.0, 1.0 );
}

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

void
idle()
{
	glutPostRedisplay();
}

int
main(int argc, char **argv)
{
    glutInit( &argc, argv );
    if( argc == 2 ) {
        volume_size_x = volume_size_y = volume_size_z = atoi( argv[1] );
    }
    else if( argc == 4 ) {
        volume_size_x = atoi( argv[1] );
        volume_size_y = atoi( argv[2] );
        volume_size_z = atoi( argv[3] );
    }
    else {
        volume_size_x = 32;
        volume_size_y = 8;
        volume_size_z = 8;
    }

    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutCreateWindow( argv[0] );
    glewInit();
    glutReshapeFunc( reshape );
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    init();
    glutMainLoop();
    return EXIT_SUCCESS;
}

double getTimeOfDay()
{
#if defined(__unix)
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
#endif
}
