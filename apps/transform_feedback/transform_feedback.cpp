/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: transform_feedback.cpp
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

// Extracting iso-surfaces from a scalar field defined in terms shader code,
// and use transform feedback to capture geometry for wireframe rendering.
//
// This example is basically the same as cayley, the main difference is that
// instead of traversing the HistoPyramid twice when rendering wireframe,
// transform feedback is used to capture the geometry in the flat shading pass,
// and the result of this is rendered again using plain OpenGL to produce the
// line rendering.
//
// The actual surface is an algebraic surface defined by
//   1 - 16xyz -4x^2  - 4y^2 - 4z^2 = iso.
// The application also provides the gradient field for this function, which is
// used instead of forward differences to determine surface normals.
//
// There are three almost identical mechanisms for transform feedback in OpenGL.
//
// The GL_NV_transform_feedback extension is slightly more cumbersome as
// varyings that will be recorded must be tagged as active before linking the
// shader program (to avoid them getting optimized away), and then the list of
// varyings to record has to be specified after linking, as a list of varying
// locations. This extension is found on all NVIDIA cards from the G80 and up.
//
// The successor, GL_EXT_transform_feedback extension simplifies the process a
// bit as one can directly specify which varyings to capture before linking,
// removing the need for the post-link step. This extension is found on
// recent AMD/ATI-cards.
//
// The OpenGL 3.0 core transform feedback is basically the same as the EXT-
// extension without the EXT-suffixes. It is found on recent NVIDIA cards, as
// the NVIDIA driver implements OpenGL 3. However, a typo in the spec with
// wrong arguments for a function have crept into GLEW, making using this
// functionality a bit troublesome.
//
// This example implements paths for both the NV and EXT extensions (determined
// by a runtime check), and should run on both NVIDIA and AMD/ATI hardware.

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/glut.h>
#endif
#include "hpmc.h"
#include "../common/common.cpp"

using std::cerr;
using std::endl;
using std::vector;
using std::string;

bool use_ext; // use EXT extension instead of NV extension

int volume_size_x;
int volume_size_y;
int volume_size_z;


GLuint mc_tri_vbo;
GLsizei mc_tri_vbo_N;

struct HPMCConstants* hpmc_c;
struct HPMCIsoSurface* hpmc_h;
struct HPMCIsoSurfaceRenderer* hpmc_th;

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
struct HPMCIsoSurfaceRenderer* hpmc_th_flat;
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
GLuint flat_f;
GLuint flat_p;
struct HPMCIsoSurfaceRenderer* hpmc_th_shaded;
std::string flat_vertex_shader =
        "varying vec3 normal;\n"
        "varying vec3 position;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 p, n;\n"
        "    extractVertex( p, n );\n"
        //   store in object coords as we use std. opengl for rendering the
        //   wireframe, which does the transform
        "    normal = n;\n"
        "    position = p;\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"
        "    gl_FrontColor = gl_Color;\n"
        "}\n";
std::string flat_fragment_shader =
        "void\n"
        "main()\n"
        "{\n"
        "    gl_FragColor = gl_Color;\n"
        "}\n";

void
init()
{
    // --- check for availability of transform feedback ------------------------
    if( GLEW_NV_transform_feedback ) {
        cerr << "Using GL_NV_transform_feedback extension." << endl;
        use_ext = false;
    }
#ifdef GL_EXT_transform_feedback
    else if( GLEW_EXT_transform_feedback ) {
        cerr << "Using GL_EXT_transform_feedback extension." << endl;
        use_ext = true;
    }
    else {
        cerr << "Neither GL_NV_transform_feedback nor "
             << "GL_EXT_transform_feedback extensions present, exiting." << endl;
        exit( EXIT_FAILURE );
    }
#else
    else {
        cerr << "Note: Compiled with old GLEW that doesn't have GL_EXT_transform_feedback defined." << endl;
        cerr << "GL_NV_transform_feedback extension missing, exiting." << endl;
        exit( EXIT_FAILURE );
    }
#endif

    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = HPMCcreateConstants( HPMC_TARGET_GL20_GLSL110, HPMC_DEBUG_STDERR );
    hpmc_h = HPMCcreateIsoSurface( hpmc_c );

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
    hpmc_th_shaded = HPMCcreateIsoSurfaceRenderer( hpmc_h );

    char *traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shaded );
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
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_shaded,
                                   shaded_p,
                                   0, 1, 2 );


    hpmc_th_flat = HPMCcreateIsoSurfaceRenderer( hpmc_h );

    traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_flat );
    const char* flat_src[2] =
    {
        traversal_code,
        flat_vertex_shader.c_str()
    };
    flat_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( flat_v, 2, &flat_src[0], NULL );
    compileShader( flat_v, "flat vertex shader" );
    free( traversal_code );

    const char* flat_fsrc[1] =
    {
        flat_fragment_shader.c_str()
    };
    flat_f = glCreateShader( GL_FRAGMENT_SHADER );
    glShaderSource( flat_f, 1, &flat_fsrc[0], NULL );
    compileShader( flat_f, "flat fragment shader" );

    // link program
    flat_p = glCreateProgram();
    glAttachShader( flat_p, flat_v );
    glAttachShader( flat_p, flat_f );

    // varyings that we record
    const char* flat_varying_names[2] =
    {
        "normal",
        "position"
    };

    // When using the EXT extension (or 3.0 core), we can directly tag varyings
    // for feedback by name before linkage. The NV extension requires that we
    // first tag the varyings as active, link the program, determine the
    // varying locations of the varyings that shall be fed back, and then ship
    // this to GL.
    if( use_ext ) {
#ifdef GL_EXT_transform_feedback
        glTransformFeedbackVaryingsEXT( flat_p,
                                        2,
                                        &flat_varying_names[0],
                                        GL_INTERLEAVED_ATTRIBS_EXT );
#endif
    }
    else {
        // tag the varyings we will record as active (so they don't get
        // optimized away).
        for( int i=0; i<2; i++) {
            glActiveVaryingNV( flat_p,
                               flat_varying_names[i] );
        }
    }

    linkProgram( flat_p, "flat program" );

    if( !use_ext ) {
        // determine the location of the varyings, and ship to GL.
        GLint varying_locs[2];
        for(int i=0; i<2; i++) {
            varying_locs[i] = glGetVaryingLocationNV( flat_p,
                                                      flat_varying_names[i] );
        }
        glTransformFeedbackVaryingsNV( flat_p,
                                     2,
                                     &varying_locs[0],
                                     GL_INTERLEAVED_ATTRIBS );
    }


    // associate program with traversal handle
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_flat,
                                   flat_p,
                                   0, 1, 2 );

    // --- set up buffer for feedback of MC triangles --------------------------
    glGenBuffers( 1, &mc_tri_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, mc_tri_vbo );
    mc_tri_vbo_N = 3*1000;
    glBufferData( GL_ARRAY_BUFFER,
                  (3+3)*mc_tri_vbo_N * sizeof(GLfloat),
                  NULL,
                  GL_DYNAMIC_COPY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

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
    HPMCbuildIsoSurface( hpmc_h, iso );

    // --- render solid surface ------------------------------------------------
    glColor3f( 0.5, 0.5, 0.5 );
    glEnable( GL_DEPTH_TEST );

    // if wireframe, do transform feedback capture
    GLsizei N = HPMCacquireNumberOfVertices( hpmc_h );
    if(!wireframe) {
        // render normally
        glColor3f( 1.0-iso, 0.0, iso );
        HPMCextractVertices( hpmc_th_shaded, GL_FALSE );
    }
    else {
        // resize buffer if needed
        if( mc_tri_vbo_N < N ) {
            mc_tri_vbo_N = static_cast<GLsizei>( 1.1f*static_cast<float>(N) );
            cerr << "resizing mc_tri_vbo to hold " << mc_tri_vbo_N << " vertices." << endl;
            glBindBuffer( GL_ARRAY_BUFFER, mc_tri_vbo );
            glBufferData( GL_ARRAY_BUFFER,
                          (3+3) * mc_tri_vbo_N * sizeof(GLfloat),
                          NULL,
                          GL_DYNAMIC_COPY );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glColor3f(0.1, 0.1, 0.2 );
        glEnable( GL_POLYGON_OFFSET_FILL );
        if( use_ext ) {
#ifdef GL_EXT_transform_feedback
            glBindBufferBaseEXT( GL_TRANSFORM_FEEDBACK_BUFFER_EXT,
                                 0, mc_tri_vbo );
            HPMCextractVerticesTransformFeedbackEXT( hpmc_th_flat, GL_FALSE );
            glFlush(); // on ATI catalyst 9.10, this is needed to avoid some artefacts
#endif
        }
        else {
            glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV,
                                0, mc_tri_vbo );
            HPMCextractVerticesTransformFeedbackNV( hpmc_th_flat, GL_FALSE );
        }
        glDisable( GL_POLYGON_OFFSET_FILL );

        // --- render wireframe ------------------------------------------------
        glUseProgram( 0 );
        glColor3f( 1.0, 1.0, 1.0 );
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );

        glBindBuffer( GL_ARRAY_BUFFER, mc_tri_vbo );
        glPushClientAttrib( GL_CLIENT_VERTEX_ARRAY_BIT );
        glInterleavedArrays( GL_N3F_V3F, 0, NULL );
        glDrawArrays( GL_TRIANGLES, 0, N );
        glPopClientAttrib();
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL );
    }
    ASSERT_GL;

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
                  N/3,
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
