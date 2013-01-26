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
#include <glhpmc/Field.hpp>

namespace resources {
    extern std::string solid_vs_110;
    extern std::string solid_vs_130;
    extern std::string solid_fs_110;
    extern std::string solid_fs_130;
    extern std::string phong_vs_110;
    extern std::string phong_vs_130;
    extern std::string phong_fs_110;
    extern std::string phong_fs_130;
    extern std::string cayley_fetch;
}

class CayleyField : public glhpmc::Field
{
public:
    CayleyField( glhpmc::HPMCConstants* constants,
                 unsigned int samples_x,
                 unsigned int samples_y,
                 unsigned int samples_z )
        : glhpmc::Field( constants, samples_x, samples_y, samples_z )
    {}

    bool
    gradients() const
    {
        return true;
    }

    const std::string
    fetcherSource(bool gradient) const
    {
        return resources::cayley_fetch;
    }

};


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
glhpmc::HPMCConstants*           hpmc_c              = NULL;
glhpmc::HPMCIsoSurface*          hpmc_h              = NULL;
glhpmc::HPMCIsoSurfaceRenderer*  hpmc_th_shaded      = NULL;
glhpmc::HPMCIsoSurfaceRenderer*  hpmc_th_flat        = NULL;
CayleyField*                    cayley_field        = NULL;


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
    hpmc_c = glhpmc::HPMCConstants::factory( hpmc_target, hpmc_debug );

    cayley_field = new CayleyField( hpmc_c,
                                    volume_size_x,
                                    volume_size_y,
                                    volume_size_z );


    hpmc_h = glhpmc::HPMCIsoSurface::factory( hpmc_c, cayley_field, binary );



    // --- phong shaded render pipeline ----------------------------------------
    {
        hpmc_th_shaded = HPMCcreateIsoSurfaceRenderer( hpmc_h );
        char* traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shaded );

        const GLchar* vs_src[2] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::phong_vs_110.c_str()
            : resources::phong_vs_130.c_str(),
            traversal_code
        };
        const GLchar* fs_src[1] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::phong_fs_110.c_str()
            : resources::phong_fs_130.c_str()
        };

        GLuint vs = glCreateShader( GL_VERTEX_SHADER );
        glShaderSource( vs, 2, vs_src, NULL );
        compileShader( vs, "shaded vertex shader" );

        GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
        glShaderSource( fs, 1, fs_src, NULL );
        compileShader( fs, "shaded fragment shader" );

        shaded_p = glCreateProgram();
        glAttachShader( shaded_p, vs );
        glAttachShader( shaded_p, fs );
        if( glhpmc::HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
            glBindFragDataLocation( shaded_p, 0, "fragment" );
        }
        linkProgram( shaded_p, "shaded program" );
        shaded_loc_pm = glGetUniformLocation( shaded_p, "PM" );
        shaded_loc_nm = glGetUniformLocation( shaded_p, "NM" );
        shaded_loc_color = glGetUniformLocation( shaded_p, "color" );

        // Associate program with traversal handle
        HPMCsetIsoSurfaceRendererProgram( hpmc_th_shaded, shaded_p, 0, 1, 2 );

        glDeleteShader( vs );
        glDeleteShader( fs );
        free( traversal_code );
    }
    // --- flat-shaded render pipeline -----------------------------------------
    {
        hpmc_th_flat = HPMCcreateIsoSurfaceRenderer( hpmc_h );
        char* traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_flat );

        const GLchar* vs_src[2] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::solid_vs_110.c_str()
            : resources::solid_vs_130.c_str(),
            traversal_code
        };
        const GLchar* fs_src[1] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::solid_fs_110.c_str()
            : resources::solid_fs_130.c_str()
        };

        GLuint vs = glCreateShader( GL_VERTEX_SHADER );
        glShaderSource( vs, 2, vs_src, NULL );
        compileShader( vs, "flat vertex shader" );

        GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
        glShaderSource( fs, 1, fs_src, NULL );
        compileShader( fs, "flat fragment shader" );

        // Program
        flat_p = glCreateProgram();
        glAttachShader( flat_p, vs );
        glAttachShader( flat_p, fs );
        if( glhpmc::HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
            glBindFragDataLocation( flat_p, 0, "fragment" );
        }
        linkProgram( flat_p, "flat program" );
        flat_loc_pm = glGetUniformLocation( flat_p, "PM" );
        flat_loc_color = glGetUniformLocation( flat_p, "color" );

        HPMCsetIsoSurfaceRendererProgram( hpmc_th_flat, flat_p, 0, 1, 2 );

        glDeleteShader( vs );
        glDeleteShader( fs );
        free( traversal_code );
    }
    glPolygonOffset( 1.0, 1.0 );
}

void
render( float t,
        float dt,
        float fps,
        const GLfloat* P,
        const GLfloat* MV,
        const GLfloat* PM,
        const GLfloat *NM,
        const GLfloat* MV_inv )
{
    // Clear screen
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Build histopyramid
    iso = sin(t);
    hpmc_h->build( iso );

    // Set up view matrices if pre 3.0
    glEnable( GL_DEPTH_TEST );
    if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf( P );
        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf( MV );
    }

    if( !wireframe ) {
        // Solid shaded rendering
        glUseProgram( shaded_p );
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 1.0-iso, 0.0, iso );
        }
        else {
            glUniformMatrix4fv( shaded_loc_pm, 1, GL_FALSE, PM );
            glUniformMatrix3fv( shaded_loc_nm, 1, GL_FALSE, NM );
            glUniform4f( shaded_loc_color,  1.0-iso, 0.0, iso, 1.f );
        }
        HPMCextractVertices( hpmc_th_shaded, GL_FALSE );
    }
    else {
        // Wireframe rendering
        glUseProgram( flat_p );
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 0.2*(1.0-iso), 0.0, 0.2*iso );
        }
        else {
            glUniformMatrix4fv( flat_loc_pm, 1, GL_FALSE, PM );
            glUniform4f( flat_loc_color,  1.0-iso, 0.0, iso, 1.f );
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable( GL_POLYGON_OFFSET_FILL );
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glDisable( GL_POLYGON_OFFSET_FILL );

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glUseProgram( flat_p );
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
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
