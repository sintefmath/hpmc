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

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include "../common/common.hpp"

using std::cerr;
using std::endl;

int                             volume_size_x       = 64;
int                             volume_size_y       = 64;
int                             volume_size_z       = 64;
float                           iso                 = 0.5f;
GLuint                          shaded_p            = 0;
GLint                           shaded_loc_pm       = -1;
GLint                           shaded_loc_nm       = -1;
GLint                           shaded_loc_color    = -1;
GLuint                          flat_p              = 0;
GLint                           flat_loc_pm         = -1;
GLint                           flat_loc_color      = -1;
struct HPMCConstants*           hpmc_c              = NULL;
struct HPMCIsoSurface*          hpmc_h              = NULL;
struct HPMCIsoSurfaceRenderer*  hpmc_th_shaded      = NULL;
struct HPMCIsoSurfaceRenderer*  hpmc_th_flat        = NULL;

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


void
printHelp( const std::string& appname )
{
    cerr << "HPMC demo application that visualizes 1-16xyz-4x^2-4y^2-4z^2=iso."<<endl<<endl;
    cerr << "Usage: " << appname << " [options] xsize [ysize zsize] "<<endl<<endl;
    cerr << "where: xsize    The number of samples in the x-direction."<<endl;
    cerr << "       ysize    The number of samples in the y-direction."<<endl;
    cerr << "       zsize    The number of samples in the z-direction."<<endl;
    cerr << "Example usage:"<<endl;
    cerr << "    " << appname << " 64"<< endl;
    cerr << "    " << appname << " 64 128 64"<< endl;
    cerr << endl;
    printOptions();
}

void
init( int argc, char** argv )
{
    if( argc > 1 ) {
        volume_size_x = atoi( argv[1] );
    }
    if( argc > 3 ) {
        volume_size_y = atoi( argv[2] );
        volume_size_z = atoi( argv[3] );
    }
    else {
        volume_size_y = volume_size_x;
        volume_size_z = volume_size_x;
    }

    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = HPMCcreateConstants( hpmc_target, hpmc_debug );
    hpmc_h = HPMCcreateIsoSurface( hpmc_c );

    if( binary ) {
        HPMCsetFieldAsBinary( hpmc_h );
    }

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
    const char* sources[2];
    char* traversal_code;

    // Shaded pipeline
    hpmc_th_shaded = HPMCcreateIsoSurfaceRenderer( hpmc_h );

    // Vertex shader
    traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shaded );

    GLuint shaded_v = glCreateShader( GL_VERTEX_SHADER );
    sources[0] = hpmc_target < HPMC_TARGET_GL30_GLSL130
               ? resources::phong_vs_110.c_str()
               : resources::phong_vs_130.c_str();
    sources[1] = traversal_code;
    glShaderSource( shaded_v, 2, sources, NULL );
    compileShader( shaded_v, "shaded vertex shader" );
    free( traversal_code );

    // Fragment shader
    GLuint shaded_f = glCreateShader( GL_FRAGMENT_SHADER );
    sources[0] = hpmc_target < HPMC_TARGET_GL30_GLSL130
               ? resources::phong_fs_110.c_str()
               : resources::phong_fs_130.c_str();
    glShaderSource( shaded_f, 1, sources, NULL );
    compileShader( shaded_f, "shaded fragment shader" );

    // Program
    shaded_p = glCreateProgram();
    glAttachShader( shaded_p, shaded_v );
    glAttachShader( shaded_p, shaded_f );
    glDeleteShader( shaded_v );
    glDeleteShader( shaded_f );
    if( HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
        glBindFragDataLocation( shaded_p, 0, "fragment" );
    }
    linkProgram( shaded_p, "shaded program" );
    shaded_loc_pm = glGetUniformLocation( shaded_p, "PM" );
    shaded_loc_nm = glGetUniformLocation( shaded_p, "NM" );
    shaded_loc_color = glGetUniformLocation( shaded_p, "color" );

    // Associate program with traversal handle
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_shaded,
                                      shaded_p,
                                      0, 1, 2 );

    // Flat-shaded pipeline
    hpmc_th_flat = HPMCcreateIsoSurfaceRenderer( hpmc_h );

    // Vertex shader
    traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_flat );
    GLuint flat_v = glCreateShader( GL_VERTEX_SHADER );
    sources[0] = hpmc_target < HPMC_TARGET_GL30_GLSL130
               ? resources::solid_vs_110.c_str()
               : resources::solid_vs_130.c_str();
    sources[1] = traversal_code;
        glShaderSource( flat_v, 2, sources, NULL );
    compileShader( flat_v, "flat vertex shader" );
    free( traversal_code );

    // Fragment shader
    GLuint flat_f = glCreateShader( GL_FRAGMENT_SHADER );
    sources[0] = hpmc_target < HPMC_TARGET_GL30_GLSL130
               ? resources::solid_fs_110.c_str()
               : resources::solid_fs_130.c_str();
    glShaderSource( flat_f, 1, sources, NULL );

    // Program
    flat_p = glCreateProgram();
    glAttachShader( flat_p, flat_v );
    glAttachShader( flat_p, flat_f );
    glDeleteShader( flat_v );
    glDeleteShader( flat_f );
    if( HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
        glBindFragDataLocation( flat_p, 0, "fragment" );
    }
    linkProgram( flat_p, "flat program" );
    flat_loc_pm = glGetUniformLocation( flat_p, "PM" );
    flat_loc_color = glGetUniformLocation( flat_p, "color" );

    // Associate program with traversal handle
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_flat,
                                      flat_p,
                                      0, 1, 2 );

    glPolygonOffset( 1.0, 1.0 );
}

void
render( float t, float dt, float fps, const GLfloat* P, const GLfloat* MV, const GLfloat* PMV, const GLfloat *NM )
{
    // Clear screen
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Build histopyramid
    iso = sin(t);
    HPMCbuildIsoSurface( hpmc_h, iso );
    // Set up view matrices if pre 3.0
    glEnable( GL_DEPTH_TEST );
    if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf( P );
        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf( MV );
    }

    if( !wireframe ) {
        // Solid shaded rendering
        glUseProgram( shaded_p );
        if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 1.0-iso, 0.0, iso );
        }
        else {
            glUniformMatrix4fv( shaded_loc_pm, 1, GL_FALSE, PMV );
            glUniformMatrix3fv( shaded_loc_nm, 1, GL_FALSE, NM );
            glUniform4f( shaded_loc_color,  1.0-iso, 0.0, iso, 1.f );
        }
        HPMCextractVertices( hpmc_th_shaded, GL_FALSE );
    }
    else {
        // Wireframe rendering
        glUseProgram( flat_p );
        if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 0.2*(1.0-iso), 0.0, 0.2*iso );
        }
        else {
            glUniformMatrix4fv( flat_loc_pm, 1, GL_FALSE, PMV );
            glUniform4f( flat_loc_color,  1.0-iso, 0.0, iso, 1.f );
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable( GL_POLYGON_OFFSET_FILL );
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glDisable( GL_POLYGON_OFFSET_FILL );

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glUseProgram( flat_p );
        if( hpmc_target < HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 1.0, 1.0, 1.0 );
        }
        else {
            glUniform4f( flat_loc_color, 1.f, 1.f, 1.f, 1.f );
        }
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}

const std::string
infoString( float fps )
{
    std::stringstream o;
    o << std::setprecision(5) << fps << " fps, "
      << volume_size_x << 'x'
      << volume_size_y << 'x'
      << volume_size_z << " samples, "
      << (int)( ((volume_size_x-1)*(volume_size_y-1)*(volume_size_z-1)*fps)/1e6 )
      << " MVPS, "
      << HPMCacquireNumberOfVertices( hpmc_h )/3
      << " vertices, iso=" << iso
      << (wireframe?"[wireframe]":"");
    return o.str();
}
