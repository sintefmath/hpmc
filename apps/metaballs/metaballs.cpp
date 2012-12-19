/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: metaballs.cpp
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

// Rendering metaballs
//
// This example demonstrates the use of a custom fetch function and how the
// application can get hold of program names to update uniform variables. In
// principle, the fetch calculates the distance field defined by eight
// metaballs, whose position is provided through uniform variables. To make
// the example more interesting, the domain is twisted time-dependently along
// the z and y-axes.

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GL/glut.h>
#include "hpmc.h"
#include "../common/common.cpp"

int volume_size_x;
int volume_size_y;
int volume_size_z;

GLuint shiny_v;
GLuint shiny_f;
GLuint shiny_p;

GLuint flat_v;
GLuint flat_p;

struct HPMCConstants* hpmc_c;
struct HPMCHistoPyramid* hpmc_h;
struct HPMCTraversalHandle* hpmc_th_shiny;
struct HPMCTraversalHandle* hpmc_th_flat;


// -----------------------------------------------------------------------------
// a metaball evaluation shader, with domain twist
std::string fetch_code =
        "uniform float twist;\n"
        "uniform vec3 centers[8];\n"
        "float\n"
        "HPMC_fetch( vec3 p )\n"
        "{\n"
        "    p = 2.0*p - 1.0;\n"
        "    float rot1 = twist*p.z;\n"
        "    float rot2 = 0.7*twist*p.y;\n"
        "    p = mat3( cos(rot1), -sin(rot1), 0,\n"
        "              sin(rot1),  cos(rot1), 0,\n"
        "                      0,          0, 1)*p;\n"
        "    p = mat3( cos(rot2), 0, -sin(rot2), \n"
        "                      0, 1,          0,\n"
        "              sin(rot2), 0,  cos(rot2) )*p;\n"
        "    p = 0.5*p + vec3(0.5);\n"
        "    float s = 0.0;\n"
        "    for(int i=0; i<8; i++) {\n"
        "        vec3 r = p-centers[i];\n"
        "        s += 0.05/dot(r,r);\n"
        "    }\n"
        "    return s;\n"
        "}\n";

// -----------------------------------------------------------------------------
// a small vertex shader that calls the provided extraction function
std::string shiny_vertex_shader =
        "varying vec3 normal;\n"
        "void\n"                                                               \
        "main()\n"                                                             \
        "{\n"                                                                  \
        "    vec3 p, n;\n"                                                     \
        "    extractVertex( p, n );\n"                                         \
        "    normal = gl_NormalMatrix * n;\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"   \
        "}\n";
std::string shiny_fragment_shader =
        "varying vec3 normal;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 v = vec3( 0.0, 0.0, 1.0 );\n"
        "    vec3 n = normalize( normal );\n"
        "    vec3 r = reflect( v, n );\n"
        "    vec3 h = 0.5*(v+n);\n"
        "    vec3 c_r = vec3(0.4, 1.3, 2.0) * max( 0.0, -r.y )\n"
        "             + vec3(0.5, 0.4, 0.2) * pow( max( 0.0, r.y), 3.0 );\n"
        "    vec3 c_s = vec3(0.7, 0.9, 1.0) * pow( max( 0.0, dot( v, h ) ), 50.0 );\n"
        "    vec3 c_f = vec3(0.8, 0.9, 1.0) * pow( 1.0-abs(n.z), 5.0 );\n"
        "    gl_FragColor = vec4( c_r + c_s + c_f, 1.0 );\n"
        "}\n";

// -----------------------------------------------------------------------------
std::string flat_vertex_shader =
        "void\n"                                                               \
        "main()\n"                                                             \
        "{\n"                                                                  \
        "    vec3 p, n;\n"                                                     \
        "    extractVertex( p, n );\n"                                         \
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"   \
        "    gl_FrontColor = gl_Color;\n"                              \
        "}\n";


// -----------------------------------------------------------------------------
void
init()
{
    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = HPMCcreateConstants();
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
                        GL_FALSE );


    // --- shiny traversal vertex shader ---------------------------------------
    hpmc_th_shiny = HPMCcreateTraversalHandle( hpmc_h );

    char *traversal_code = HPMCgetTraversalShaderFunctions( hpmc_th_shiny );
    const char* shiny_vsrc[2] =
    {
        traversal_code,
        shiny_vertex_shader.c_str()
    };
    shiny_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( shiny_v, 2, &shiny_vsrc[0], NULL );
    compileShader( shiny_v, "shiny vertex shader" );
    free( traversal_code );

    const char* shiny_fsrc[1] =
    {
        shiny_fragment_shader.c_str()
    };
    shiny_f = glCreateShader( GL_FRAGMENT_SHADER );
    glShaderSource( shiny_f, 1, &shiny_fsrc[0], NULL );
    compileShader( shiny_f, "shiny fragment shader" );

    shiny_p = glCreateProgram();
    glAttachShader( shiny_p, shiny_v );
    glAttachShader( shiny_p, shiny_f );
    linkProgram( shiny_p, "shiny program" );

    // associate the linked program with the traversal handle
    HPMCsetTraversalHandleProgram( hpmc_th_shiny,
                                   shiny_p,
                                   0, 1, 2 );

    // --- flat traversal vertex shader ----------------------------------------
    hpmc_th_flat = HPMCcreateTraversalHandle( hpmc_h );

    traversal_code = HPMCgetTraversalShaderFunctions( hpmc_th_shiny );
    const char* flat_src[2] =
    {
        traversal_code,
        flat_vertex_shader.c_str()
    };
    flat_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( flat_v, 2, &flat_src[0], NULL );
    compileShader( flat_v, "flat vertex shader" );
    free( traversal_code );

    flat_p = glCreateProgram();
    glAttachShader( flat_p, flat_v );
    linkProgram( flat_p, "flat program" );

    // associate the linked program with the traversal handle
    HPMCsetTraversalHandleProgram( hpmc_th_flat,
                                   flat_p,
                                   0, 1, 2 );



    glPolygonOffset( 1.0, 1.0 );
}

// -----------------------------------------------------------------------------
void
render( float t, float dt, float fps )
{
/*
void
display()
{
    ASSERT_GL;

    // --- timing --------------------------------------------------------------
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
*/
    // --- clear screen and set up view ----------------------------------------
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -0.1*aspect_x, 0.1*aspect_x, -0.1*aspect_y, 0.1*aspect_y, 0.5, 3.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0.0f, 0.0f, -2.0f );
    glRotatef( 20, 1.0, 0.0, 0.0 );
    glRotatef( 20.0*t, 0.0, 1.0, 0.0 );
    glTranslatef( -0.5f, -0.5f, -0.5f );

    // --- update metaballs position -------------------------------------------
    std::vector<GLfloat> centers( 3*8 );
    for( size_t i=0; i<8; i++ ) {
        centers[3*i+0] = 0.5+0.3*sin( t+sin(0.1*t)*i );
        centers[3*i+1] = 0.5+0.3*cos( 0.9*t+sin(0.1*t)*i );
        centers[3*i+2] = 0.5+0.3*cos( 0.7*t+sin(0.01*t)*i );
    }
    float twist = 5.0*sin(0.1*t);

    GLuint builder = HPMCgetBuilderProgram( hpmc_h );

    glUseProgram( builder );
    glUniform1f( glGetUniformLocation( builder, "twist" ), twist );
    glUniform3fv( glGetUniformLocation( builder, "centers" ), 8, &centers[0] );

    glUseProgram( shiny_p );
    glUniform1f( glGetUniformLocation( shiny_p, "twist" ), twist );
    glUniform3fv( glGetUniformLocation( shiny_p, "centers" ), 8, &centers[0] );

    glUseProgram( flat_p );
    glUniform1f( glGetUniformLocation( flat_p, "twist" ), twist );
    glUniform3fv( glGetUniformLocation( flat_p, "centers" ), 8, &centers[0] );

    glUseProgram( 0 );

    // --- build HistoPyramid --------------------------------------------------
    GLfloat iso = 10.0f;
    HPMCbuildHistopyramid( hpmc_h, iso );

    // --- render solid surface ------------------------------------------------
    glEnable( GL_DEPTH_TEST );

    if( !wireframe ) {
        HPMCextractVertices( hpmc_th_shiny );
    }
    else {
        glColor3f( 0.1, 0.1, 0.5 );

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
    glUseProgram( 0 );
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
    glutReshapeFunc( reshape );
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    init();
    glutMainLoop();
    return EXIT_SUCCESS;
}
