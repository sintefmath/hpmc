/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: cayley.cpp
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

// Extracting iso-surfaces from a scalar field defined in terms shader code.
//
// This example is similar to the texture3d example. The difference is that
// instead of fetching scalar values from a texture 3d, the scalar value is
// fetched from an application provided shader function, which in this example
// evaluates the algebraic surface defined by
//   1 - 16xyz -4x^2  - 4y^2 - 4z^2 = iso.
// The application also provides the gradient field for this function, which is
// used instead of forward differences to determine surface normals.
//
// For each frame, a time-depenent iso-value is calculated, passed to HPMC which
// analyzes the scalar field using this iso-value. Then, HPMC renders the
// corresponding iso-surface. Wireframe rendering is done straight-forwardly by
// rendering the surface twice (traversing the HistoPyramid both times), one
// time with solid triangles in a dark color offset slightly away from the
// camera, and the second time using the line-drawing polygon mode to render
// the actual wireframe in white.

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#endif
#include "hpmc.h"
#include "../common/common.cpp"

#ifdef __unix
#include <sys/time.h>
#include <unistd.h>
#endif

int volume_size_x;
int volume_size_y;
int volume_size_z;

struct HPMCConstants* hpmc_c;
struct HPMCHistoPyramid* hpmc_h;

// -----------------------------------------------------------------------------
std::string fetch_code =
        // evaluates the scalar field
        "float\n"
        "HPMC_fetch( vec3 p )\n"
        "{\n"
        "    p *= 2.0;\n"
        "    p -= 1.0;\n"
        "    return 1.0 - 16.0*p.x*p.y*p.z - 4.0*p.x*p.x - 4.0*p.y*p.y - 4.0*p.z*p.z;\n"
        "}\n"
        "vec4\n"
        // evaluates the gradient as well as the scalar field
        "HPMC_fetchGrad( vec3 p )\n"
        "{\n"
        "    p *= 2.0;\n"
        "    p -= 1.0;\n"
        "    return vec4( -16.0*p.y*p.z - 8.0*p.x,\n"
        "                 -16.0*p.x*p.z - 8.0*p.y,\n"
        "                 -16.0*p.x*p.y - 8.0*p.z,\n"
        "                 1.0 - 16.0*p.x*p.y*p.z - 4.0*p.x*p.x - 4.0*p.y*p.y - 4.0*p.z*p.z );\n"
        "}\n";


// -----------------------------------------------------------------------------
GLuint shaded_v;
GLuint shaded_f;
GLuint shaded_p;
struct HPMCTraversalHandle* hpmc_th_flat;
std::string shaded_vertex_shader =
        "varying vec3 normal;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 p, n;\n"
        "    extractVertex( p, n );\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"
        "    normal = gl_NormalMatrix * n;\n"
        "    gl_FrontColor = gl_Color;\n"
        "}\n";
std::string shaded_fragment_shader =
        "varying vec3 normal;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    const vec3 v = vec3( 0.0, 0.0, 1.0 );\n"
        "    vec3 l = normalize( vec3( 1.0, 1.0, 1.0 ) );\n"
        "    vec3 h = normalize( v+l );\n"
        "    vec3 n = normalize( normal );\n"
        "    float diff = max( 0.1, dot( n, l ) );\n"
        "    float spec = pow( max( 0.0, dot(n, h)), 20.0);\n"
        "    gl_FragColor = diff * gl_Color +\n"
        "                   spec * vec4(1.0);\n"
        "}\n";

// -----------------------------------------------------------------------------
GLuint flat_v;
GLuint flat_p;
struct HPMCTraversalHandle* hpmc_th_shaded;
std::string flat_vertex_shader =
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 p, n;\n"
        "    extractVertex( p, n );\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"
        "    gl_FrontColor = gl_Color;\n"
        "}\n";

// -----------------------------------------------------------------------------
void
init()
{
    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = HPMCcreateConstants( 4, 3 );
    hpmc_h = HPMCcreateHistoPyramid( hpmc_c );

    HPMCsetLatticeSize( hpmc_h,
                        volume_size_x,
                        volume_size_y,
                        volume_size_z );

    HPMCsetGridSize( hpmc_h,
                     volume_size_x-1,
                     volume_size_y-1,
                     volume_size_z-1 );

    HPMCsetGridExtent( hpmc_h,
                       1.0f,
                       1.0f,
                       1.0f );

    HPMCsetFieldCustom( hpmc_h,
                        fetch_code.c_str(),
                        0,
                        GL_TRUE );



     // --- create traversal vertex shader --------------------------------------
    hpmc_th_shaded = HPMCcreateTraversalHandle( hpmc_h );

    char *traversal_code = HPMCgetTraversalShaderFunctions( hpmc_th_shaded );
    const char* shaded_vsrc[2] =
    {
        traversal_code,
        shaded_vertex_shader.c_str()
    };
    shaded_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( shaded_v, 2, &shaded_vsrc[0], NULL );
    compileShader( shaded_v, "shaded vertex shader" );
    free( traversal_code );

    const char* shaded_fsrc[1] =
    {
        shaded_fragment_shader.c_str()
    };
    shaded_f = glCreateShader( GL_FRAGMENT_SHADER );
    glShaderSource( shaded_f, 1, &shaded_fsrc[0], NULL );
    compileShader( shaded_f, "shaded fragment shader" );

    // link program
    shaded_p = glCreateProgram();
    glAttachShader( shaded_p, shaded_v );
    glAttachShader( shaded_p, shaded_f );
    linkProgram( shaded_p, "shaded program" );

    // associate program with traversal handle
    HPMCsetTraversalHandleProgram( hpmc_th_shaded,
                                   shaded_p,
                                   0, 1, 2 );

    hpmc_th_flat = HPMCcreateTraversalHandle( hpmc_h );

    traversal_code = HPMCgetTraversalShaderFunctions( hpmc_th_flat );
    const char* flat_src[2] =
    {
        traversal_code,
        flat_vertex_shader.c_str()
    };
    flat_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( flat_v, 2, &flat_src[0], NULL );
    compileShader( flat_v, "flat vertex shader" );
    free( traversal_code );

    // link program
    flat_p = glCreateProgram();
    glAttachShader( flat_p, flat_v );
    linkProgram( flat_p, "flat program" );

    // associate program with traversal handle
    HPMCsetTraversalHandleProgram( hpmc_th_flat,
                                   flat_p,
                                   0, 1, 2 );

    glPolygonOffset( 1.0, 1.0 );
}

// -----------------------------------------------------------------------------
void
render( float t, float dt, float fps )
{
    ASSERT_GL;

    // --- clear screen and set up view ----------------------------------------
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -0.2*aspect_x, 0.2*aspect_x, -0.2*aspect_y, 0.2*aspect_y, 0.5, 3.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0.0f, 0.0f, -2.0f );
    glRotatef( 20, 1.0, 0.0, 0.0 );
    glRotatef( 20.0*t, 0.0, 1.0, 0.0 );
    glTranslatef( -0.5f, -0.5f, -0.5f );

    // --- build HistoPyramid --------------------------------------------------
    float iso = sin(t);
    HPMCbuildHistopyramid( hpmc_h, iso );

    // --- render solid surface ------------------------------------------------
    glEnable( GL_DEPTH_TEST );
    if( !wireframe ) {
        glColor3f( 1.0-iso, 0.0, iso );
        HPMCextractVertices( hpmc_th_shaded );
    }
    else {
        glColor3f( 0.2*(1.0-iso), 0.0, 0.2*iso );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable( GL_POLYGON_OFFSET_FILL );
        HPMCextractVertices( hpmc_th_flat );
        glDisable( GL_POLYGON_OFFSET_FILL );

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glColor3f( 1.0, 1.0, 1.0 );
        HPMCextractVertices( hpmc_th_flat );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    // --- render text string --------------------------------------------------
    static char message[512] = "";
    if( floor(5.0*(t-dt)) != floor(5.0*(t)) ) {
        snprintf( message, 512,
                  "%.1f fps, %dx%dx%d samples, %d mvps, %d triangles, iso=%.2f%s",
                  fps,
                  volume_size_x,
                  volume_size_y,
                  volume_size_z,
                  (int)( ((volume_size_x-1)*(volume_size_y-1)*(volume_size_z-1)*fps)/1e6 ),
                  HPMCacquireNumberOfVertices( hpmc_h )/3,
                  iso,
                  wireframe ? " [wireframe]" : "");
    }
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glDisable( GL_DEPTH_TEST );
    glColor3f( 1.0, 1.0, 1.0 );
    glRasterPos2f( -0.99, 0.95 );
    for(int i=0; i<255 && message[i] != '\0'; i++) {
        glutBitmapCharacter( GLUT_BITMAP_8_BY_13, (int)message[i] );
    }
}

// -----------------------------------------------------------------------------
int
main(int argc, char **argv)
{
    glutInit( &argc, argv );
#ifdef DEBUG
    glutInitContextFlags( GLUT_DEBUG );
    glewExperimental = GL_TRUE;
#endif
    if( argc == 2 ) {
        volume_size_x = volume_size_y = volume_size_z = atoi( argv[1] );
    }
    else if( argc == 4 ) {
        volume_size_x = atoi( argv[1] );
        volume_size_y = atoi( argv[2] );
        volume_size_z = atoi( argv[3] );
    }
    else {
        volume_size_x = 64;
        volume_size_y = 64;
        volume_size_z = 64;
    }
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutInitWindowSize( 1280, 720 );
    glutCreateWindow( argv[0] );
    glewInit();
    setupGLDebug();
    glutReshapeFunc( reshape );
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    init();
    glutMainLoop();
    return EXIT_SUCCESS;
}
