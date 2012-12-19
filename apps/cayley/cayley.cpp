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
/*
#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/glut.h>
#endif
*/
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include "hpmc.h"
#include "../common/common.cpp"

#ifdef __unix
#include <sys/time.h>
#include <unistd.h>
#endif

int volume_size_x;
int volume_size_y;
int volume_size_z;

HPMCTarget hpmc_target = HPMC_TARGET_GL20_GLSL110;
HPMCDebugBehaviour hpmc_debug = HPMC_DEBUG_KHR_DEBUG;

struct HPMCConstants* hpmc_c;
struct HPMCIsoSurface* hpmc_h;

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
GLint  shaded_loc_pm;
GLint  shaded_loc_nm;
GLint  shaded_loc_color;
struct HPMCIsoSurfaceRenderer* hpmc_th_flat;
std::string shader_version_130 =
        "#version 130\n";

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

std::string shaded_vertex_shader_130 =
        "out vec3 normal;\n"
        "uniform mat4 PM;\n"
        "uniform mat3 NM;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 p, n;\n"
        "    extractVertex( p, n );\n"
        "    gl_Position = PM * vec4( p, 1.0 );\n"
        "    normal = NM * n;\n"
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
        "    gl_FragColor = diff * gl_Color + spec * vec4(1.0);\n"
        "}\n";
std::string shaded_fragment_shader_130 =
        "in vec3 normal;\n"
        "out vec4 fragment;\n"
        "uniform vec4 color;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    const vec3 v = vec3( 0.0, 0.0, 1.0 );\n"
        "    vec3 l = normalize( vec3( 1.0, 1.0, 1.0 ) );\n"
        "    vec3 h = normalize( v+l );\n"
        "    vec3 n = normalize( normal );\n"
        "    float diff = max( 0.1, dot( n, l ) );\n"
        "    float spec = pow( max( 0.0, dot(n, h)), 20.0);\n"
        "    fragment = diff * color + spec * vec4(1.0);\n"
        "}\n";

// -----------------------------------------------------------------------------
GLuint flat_v;
GLuint flat_f;
GLuint flat_p;
GLint  flat_loc_pm;
GLint  flat_loc_color;

struct HPMCIsoSurfaceRenderer* hpmc_th_shaded;
std::string flat_vertex_shader =
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 p, n;\n"
        "    extractVertex( p, n );\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );\n"
        "    gl_FrontColor = gl_Color;\n"
        "}\n";
std::string flat_vertex_shader_130 =
        "uniform mat4 PM;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    vec3 p, n;\n"
        "    extractVertex( p, n );\n"
        "    gl_Position = PM * vec4( p, 1.0 );\n"
        "    gl_FrontColor = gl_Color;\n"
        "}\n";
std::string flat_fragment_shader =
        "void\n"
        "main()\n"
        "{\n"
        "    gl_FragColor = gl_Color;\n"
        "}\n";
std::string flat_fragment_shader_130 =
        "out vec4 fragment;\n"
        "uniform vec4 color;\n"
        "void\n"
        "main()\n"
        "{\n"
        "    fragment = color;\n"
        "}\n";


// -----------------------------------------------------------------------------
void
init()
{
    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = HPMCcreateConstants( hpmc_target, hpmc_debug );
    hpmc_h = HPMCcreateIsoSurface( hpmc_c );

    HPMCsetFieldAsBinary( hpmc_h );

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

    const char* sources[3];
     // --- create traversal vertex shader --------------------------------------
    hpmc_th_shaded = HPMCcreateIsoSurfaceRenderer( hpmc_h );
    char *traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shaded );

    shaded_v = glCreateShader( GL_VERTEX_SHADER );
    if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
        sources[0] = traversal_code;
        sources[1] = shaded_vertex_shader.c_str();
        glShaderSource( shaded_v, 2, sources, NULL );
    }
    else {
        sources[0] = shader_version_130.c_str();
        sources[1] = traversal_code;
        sources[2] =  shaded_vertex_shader_130.c_str();
        glShaderSource( shaded_v, 3, sources, NULL );
    }
    compileShader( shaded_v, "shaded vertex shader" );
    free( traversal_code );

    shaded_f = glCreateShader( GL_FRAGMENT_SHADER );
    if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
        sources[0] = shaded_fragment_shader.c_str();
        glShaderSource( shaded_f, 1, sources, NULL );
    }
    else {
        sources[0] = shader_version_130.c_str();
        sources[1] = shaded_fragment_shader_130.c_str();
        glShaderSource( shaded_f, 2, sources, NULL );
    }
    compileShader( shaded_f, "shaded fragment shader" );





    // link program
    shaded_p = glCreateProgram();
    glAttachShader( shaded_p, shaded_v );
    glAttachShader( shaded_p, shaded_f );
    if( HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
        glBindFragDataLocation( shaded_p, 0, "fragment" );
    }
    linkProgram( shaded_p, "shaded program" );
    shaded_loc_pm = glGetUniformLocation( shaded_p, "PM" );
    shaded_loc_nm = glGetUniformLocation( shaded_p, "NM" );
    shaded_loc_color = glGetUniformLocation( shaded_p, "color" );
    // associate program with traversal handle
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_shaded,
                                   shaded_p,
                                   0, 1, 2 );

    hpmc_th_flat = HPMCcreateIsoSurfaceRenderer( hpmc_h );


    traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_flat );
    flat_v = glCreateShader( GL_VERTEX_SHADER );
    if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
        sources[0] = traversal_code;
        sources[1] = flat_vertex_shader.c_str();
        glShaderSource( flat_v, 2, sources, NULL );
    }
    else {
        sources[0] = shader_version_130.c_str();
        sources[1] = traversal_code;
        sources[2] = flat_vertex_shader_130.c_str();
        glShaderSource( flat_v, 3, sources, NULL );
    }
    compileShader( flat_v, "flat vertex shader" );
    free( traversal_code );
    flat_f = glCreateShader( GL_FRAGMENT_SHADER );
    if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
        sources[0] = flat_fragment_shader.c_str();
        glShaderSource( flat_f, 1, sources, NULL );
    }
    else {
        sources[0] = shader_version_130.c_str();
        sources[1] = flat_fragment_shader_130.c_str();
        glShaderSource( flat_f, 2, sources, NULL );
    }

    // link program
    flat_p = glCreateProgram();
    glAttachShader( flat_p, flat_v );
    glAttachShader( flat_p, flat_f );
    if( HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
        glBindFragDataLocation( flat_p, 0, "fragment" );
    }
    linkProgram( flat_p, "flat program" );
    flat_loc_pm = glGetUniformLocation( flat_p, "PM" );
    flat_loc_color = glGetUniformLocation( flat_p, "color" );

    // associate program with traversal handle
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_flat,
                                   flat_p,
                                   0, 1, 2 );

    glPolygonOffset( 1.0, 1.0 );
    ASSERT_GL;
}

void
frustum( GLfloat* dst, GLfloat l, GLfloat r, GLfloat b, GLfloat t, GLfloat n, GLfloat f )
{
    dst[ 0] = 2.f*n/(r-l); dst[ 1] = 0.f;         dst[ 2] = 0.f;            dst[ 3] = 0.f;
    dst[ 4] = 0.f;         dst[ 5] = 2.f*n/(t-b); dst[ 6] = 0.f;            dst[ 7] = 0.f;
    dst[ 8] = (r+l)/(r-l); dst[ 9] = (t+b)/(t-b); dst[10] = -(f+n)/(f-n);   dst[11] = -1.f;
    dst[12] = 0.f;         dst[13] = 0.f;         dst[14] = -2.f*f*n/(f-n); dst[15] = 0.f;
}

void
translate( GLfloat* dst, GLfloat x, GLfloat y, GLfloat z )
{
    dst[ 0] = 1.f;    dst[ 1] = 0.f;    dst[ 2] = 0.f;    dst[ 3] = 0.f;
    dst[ 4] = 0.f;    dst[ 5] = 1.f;    dst[ 6] = 0.f;    dst[ 7] = 0.f;
    dst[ 8] = 0.f;    dst[ 9] = 0.f;    dst[10] = 1.f;    dst[11] = 0.f;
    dst[12] = x;      dst[13] = y;      dst[14] = z;      dst[15] = 1.f;
}

void
rotX( GLfloat* dst, GLfloat degrees )
{
    GLfloat c = cosf( (M_PI/180.f)*degrees );
    GLfloat s = sinf( (M_PI/180.f)*degrees );
    dst[ 0] = 1.f;    dst[ 1] = 0.f;    dst[ 2] = 0.f;    dst[ 3] = 0.f;
    dst[ 4] = 0.f;    dst[ 5] = c;      dst[ 6] = s;      dst[ 7] = 0.f;
    dst[ 8] = 0.f;    dst[ 9] = -s;     dst[10] = c;      dst[11] = 0.f;
    dst[12] = 0.f;    dst[13] = 0.f;    dst[14] = 0.f;    dst[15] = 1.f;
}

void
rotY( GLfloat* dst, GLfloat degrees )
{
    GLfloat c = cosf( (M_PI/180.f)*degrees );
    GLfloat s = sinf( (M_PI/180.f)*degrees );
    dst[ 0] = c;      dst[ 1] = 0.f;    dst[ 2] = -s;     dst[ 3] = 0.f;
    dst[ 4] = 0.f;    dst[ 5] = 1.f;    dst[ 6] = 0.f;    dst[ 7] = 0.f;
    dst[ 8] = s;      dst[ 9] = 0.f;    dst[10] = c;      dst[11] = 0.f;
    dst[12] = 0.f;    dst[13] = 0.f;    dst[14] = 0.f;    dst[15] = 1.f;
}

void
extractUpperLeft3x3( GLfloat* dst, GLfloat* src )
{
    for(int j=0; j<3; j++ ) {
        for(int i=0; i<3; i++ ) {
            dst[3*j+i] = src[4*j+i];
        }
    }
}

void
rightMulAssign( GLfloat* A, GLfloat* B )
{
    for(int j=0; j<4; j++ ) {
        GLfloat row[4];
        for(int i=0; i<4; i++ ) {
            row[i] = 0.f;
            for(int k=0; k<4; k++) {
                row[i] += A[ 4*k + j ]*B[ 4*i + k ];
            }
        }
        for(int k=0; k<4; k++) {
            A[4*k+j] = row[k];
        }
    }
}

// -----------------------------------------------------------------------------
void
render( float t, float dt, float fps )
{
    GLfloat PM[16];
    GLfloat NM[9];

    // --- clear screen and set up view ----------------------------------------
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glFrustum( -0.2*aspect_x, 0.2*aspect_x, -0.2*aspect_y, 0.2*aspect_y, 0.5, 3.0 );

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
        glTranslatef( 0.0f, 0.0f, -2.0f );
        glRotatef( 20, 1.0, 0.0, 0.0 );
        glRotatef( 20.0*t, 0.0, 1.0, 0.0 );
        glTranslatef( -0.5f, -0.5f, -0.5f );
    }
    else {
        GLfloat M[16];
        GLfloat tmp[16];
        translate( M, 0.f, 0.f, -2.f );
        rotX( tmp, 20.f );
        rightMulAssign( M, tmp );
        rotY( tmp, 20.f*t );
        rightMulAssign( M, tmp );
        translate( tmp, -0.5f, -0.5f, -0.5f );
        rightMulAssign( M, tmp );
        extractUpperLeft3x3( NM, M );

        frustum( PM,  -0.2*aspect_x, 0.2*aspect_x, -0.2*aspect_y, 0.2*aspect_y, 0.5, 3.0 );
        rightMulAssign( PM, M );
    }

    // --- build HistoPyramid --------------------------------------------------
    float iso = sin(t);
    HPMCbuildIsoSurface( hpmc_h, iso );

    // --- render solid surface ------------------------------------------------
    glEnable( GL_DEPTH_TEST );
    if( !wireframe ) {
        if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 1.0-iso, 0.0, iso );
        }
        else {
            glUseProgram( shaded_p );
            glUniformMatrix4fv( shaded_loc_pm, 1, GL_FALSE, PM );
            glUniformMatrix3fv( shaded_loc_nm, 1, GL_FALSE, NM );
            glUniform4f( shaded_loc_color,  1.0-iso, 0.0, iso, 1.f );
        }
        HPMCextractVertices( hpmc_th_shaded, GL_FALSE );
    }
    else {
        if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 0.2*(1.0-iso), 0.0, 0.2*iso );
        }
        else {
            glUseProgram( flat_p );
            glUniformMatrix4fv( flat_loc_pm, 1, GL_FALSE, PM );
            glUniform4f( flat_loc_color,  1.0-iso, 0.0, iso, 1.f );
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable( GL_POLYGON_OFFSET_FILL );
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glDisable( GL_POLYGON_OFFSET_FILL );

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 1.0, 1.0, 1.0 );
        }
        else {
            glUseProgram( flat_p );
            glUniform4f( flat_loc_color, 1.f, 1.f, 1.f, 1.f );
        }
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
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

        if( HPMC_TARGET_GL31_GLSL140 < hpmc_target ) {
            fprintf( stderr, "%s\n", message );
        }
    }
    if( hpmc_target <= HPMC_TARGET_GL31_GLSL140 ) {
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
    //abort();
}

// -----------------------------------------------------------------------------
#ifndef APIENTRY
#define APIENTRY
#endif

static void APIENTRY debugLogger( GLenum source,
                                  GLenum type,
                                  GLuint id,
                                  GLenum severity,
                                  GLsizei length,
                                  const GLchar* message,
                                  void* data )
{
    const char* source_str = "---";
    switch( source ) {
    case GL_DEBUG_SOURCE_API: source_str = "API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM: source_str = "WSY"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: source_str = "SCM"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY: source_str = "3PY"; break;
    case GL_DEBUG_SOURCE_APPLICATION: source_str = "APP"; break;
    case GL_DEBUG_SOURCE_OTHER: source_str = "OTH"; break;
    }

    const char* type_str = "---";
    switch( type ) {
    case GL_DEBUG_TYPE_ERROR: type_str = "error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type_str = "deprecated"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: type_str = "undef"; break;
    case GL_DEBUG_TYPE_PORTABILITY: type_str = "portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE: type_str = "performance"; break;
    case GL_DEBUG_TYPE_OTHER: type_str = "other"; break;
    }

    const char* severity_str = "---";
    switch( severity ) {
    case GL_DEBUG_SEVERITY_HIGH: severity_str = "high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM: severity_str = "medium"; break;
    case GL_DEBUG_SEVERITY_LOW: severity_str = "low"; break;
    }

    fprintf( stderr, "GL debug [src=%s, type=%s, severity=%s]: %s\n",
             source_str,
             type_str,
             severity_str,
             message );

}

static void APIENTRY debugLoggerARB( GLenum source,
                                     GLenum type,
                                     GLenum id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     void* data )
{
    const char* source_str = "---";
    switch( source ) {
    case GL_DEBUG_SOURCE_API_ARB: source_str = "API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB: source_str = "WSY"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB: source_str = "SCM"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY_ARB: source_str = "3PY"; break;
    case GL_DEBUG_SOURCE_APPLICATION_ARB: source_str = "APP"; break;
    case GL_DEBUG_SOURCE_OTHER_ARB: source_str = "OTH"; break;
    }

    const char* type_str = "---";
    switch( type ) {
    case GL_DEBUG_TYPE_ERROR_ARB: type_str = "error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB: type_str = "deprecated"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB: type_str = "undef"; break;
    case GL_DEBUG_TYPE_PORTABILITY_ARB: type_str = "portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE_ARB: type_str = "performance"; break;
    case GL_DEBUG_TYPE_OTHER_ARB: type_str = "other"; break;
    }

    const char* severity_str = "---";
    switch( severity ) {
    case GL_DEBUG_SEVERITY_HIGH: severity_str = "high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM: severity_str = "medium"; break;
    case GL_DEBUG_SEVERITY_LOW: severity_str = "low"; break;
    }

    fprintf( stderr, "GL debug [src=%s, type=%s, severity=%s]: %s\n",
             source_str,
             type_str,
             severity_str,
             message );

}

// -----------------------------------------------------------------------------
int
main(int argc, char **argv)
{
    int dim_n = 0;
    int dim[3] = { 64, 64, 64 };
    glutInit( &argc, argv );


    for( int i=1; i<argc; i++ ) {
        if( strcmp( argv[i], "--target-gl20" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL20_GLSL110;
        }
        else if( strcmp( argv[i], "--target-gl21" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL21_GLSL120;
        }
        else if( strcmp( argv[i], "--target-gl30" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL30_GLSL130;
        }
        else if( strcmp( argv[i], "--target-gl31" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL31_GLSL140;
        }
        else if( strcmp( argv[i], "--target-gl32" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL32_GLSL150;
        }
        else if( strcmp( argv[i], "--target-gl33" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL33_GLSL330;
        }
        else if( strcmp( argv[i], "--target-gl40" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL40_GLSL400;
        }
        else if( strcmp( argv[i], "--target-gl41" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL41_GLSL410;
        }
        else if( strcmp( argv[i], "--target-gl42" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL42_GLSL420;
        }
        else if( strcmp( argv[i], "--target-gl43" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL43_GLSL430;
        }
        else if( strcmp( argv[i], "--debug-none" ) == 0 ) {
            hpmc_debug = HPMC_DEBUG_NONE;
        }
        else if( strcmp( argv[i], "--debug-stderr" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_STDERR;
        }
        else if( strcmp( argv[i], "--debug-stderr-verbose" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_STDERR_VERBOSE;
        }
        else if( strcmp( argv[i], "--debug-khr-debug" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_KHR_DEBUG;
        }
        else if( strcmp( argv[i], "--debug-khr-debug-verbose" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_KHR_DEBUG_VERBOSE;
        }
        else if( strcmp( argv[i], "--help" ) == 0 ) {
            fprintf( stderr, "Usage: [target] [dim]\n" );
            fprintf( stderr, "  --target-gl20\n" );
            fprintf( stderr, "  --target-gl21\n" );
            fprintf( stderr, "  --target-gl30\n" );
            fprintf( stderr, "  --target-gl31\n" );
            fprintf( stderr, "  --target-gl32\n" );
            fprintf( stderr, "  --target-gl33\n" );
            fprintf( stderr, "  --target-gl40\n" );
            fprintf( stderr, "  --target-gl41\n" );
            fprintf( stderr, "  --target-gl42\n" );
            fprintf( stderr, "  --debug-none\n" );
            fprintf( stderr, "  --debug-stderr\n" );
            fprintf( stderr, "  --debug-stderr-verbose\n" );
            fprintf( stderr, "  --debug-khr-debug\n" );
            fprintf( stderr, "  --debug-khr-debug-verbose\n" );
            exit( -1 );
        }
        else if (dim_n < 3 ) {
            dim[ dim_n++ ] = atoi( argv[i] );
        }
    }



    if( dim_n < 3 ) {
        dim[1] = dim[0];
        dim[2] = dim[0];
    }
    volume_size_x = dim[0];
    volume_size_y = dim[1];
    volume_size_z = dim[2];

    switch( hpmc_target ) {
    case HPMC_TARGET_GL20_GLSL110:
        glutInitContextVersion( 2, 0 );
        break;
    case HPMC_TARGET_GL21_GLSL120:
        glutInitContextVersion( 2, 1 );
        break;
    case HPMC_TARGET_GL30_GLSL130:
        glutInitContextVersion( 3, 0 );
        break;
    case HPMC_TARGET_GL31_GLSL140:
        glutInitContextVersion( 3, 1 );
        break;
    case HPMC_TARGET_GL32_GLSL150:
        glutInitContextVersion( 3, 2 );
        break;
    case HPMC_TARGET_GL33_GLSL330:
        glutInitContextVersion( 3, 3 );
        break;
    case HPMC_TARGET_GL40_GLSL400:
        glutInitContextVersion( 4, 0 );
        break;
    case HPMC_TARGET_GL41_GLSL410:
        glutInitContextVersion( 4, 1 );
        break;
    case HPMC_TARGET_GL42_GLSL420:
        glutInitContextVersion( 4, 2 );
        break;
    case HPMC_TARGET_GL43_GLSL430:
        glutInitContextVersion( 4, 2 );
        break;
    }

    glutInitContextFlags( GLUT_CORE_PROFILE | GLUT_DEBUG );
/*                          ((hpmc_debug == HPMC_DEBUG_KHR_DEBUG) ||
                           (hpmc_debug == HPMC_DEBUG_KHR_DEBUG_VERBOSE)
                           ? GLUT_DEBUG : 0 )
                          );
*/
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutInitWindowSize( 1280, 720 );
    glutCreateWindow( argv[0] );
    GLenum error = glGetError();
    while( error != GL_NO_ERROR ) {
        fprintf( stderr, "Context creation created GL error %x\n", error );
        error = glGetError();
    }
    glewExperimental = GL_TRUE;
    GLenum glew_error = glewInit();
    if( glew_error != GLEW_OK ) {
        fprintf( stderr, "GLEW failed to initialize, exiting.\n" );
        abort();
    }
    error = glGetError();
    while( error != GL_NO_ERROR ) {
        fprintf( stderr, "GLEW initialization created GL error %x\n", error );
        error = glGetError();
    }
    if( (hpmc_debug == HPMC_DEBUG_KHR_DEBUG) || (hpmc_debug == HPMC_DEBUG_KHR_DEBUG_VERBOSE) ) {
        if( glewIsSupported( "GL_KHR_debug" ) ) {
            glEnable( GL_DEBUG_OUTPUT_SYNCHRONOUS );
            glDebugMessageCallback( debugLogger, NULL );
            glDebugMessageControl( GL_DONT_CARE,
                                   GL_DONT_CARE,
                                   hpmc_debug == HPMC_DEBUG_KHR_DEBUG_VERBOSE ? GL_DEBUG_SEVERITY_LOW : GL_DEBUG_SEVERITY_MEDIUM,
                                   0, NULL, GL_TRUE );
        }
        else {
            fprintf( stderr, "GL_KHR_debug extension not present.\n" );
        }
    }
    error = glGetError();
    while( error != GL_NO_ERROR ) {
        fprintf( stderr, "Debug setup created GL error %x\n", error );
        error = glGetError();
    }

    glutReshapeFunc( reshape );
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    init();
    glutMainLoop();
    return EXIT_SUCCESS;
}
