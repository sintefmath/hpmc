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
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/glut.h>
#endif
#include <glhpmc/glhpmc.hpp>
#include <glhpmc/Field.hpp>
#include "../common/common.hpp"

namespace resources {
    extern std::string    solid_vs_110;
    extern std::string    solid_vs_130;
    extern std::string    solid_fs_130;
    extern std::string    solid_fs_110;
    extern std::string    shiny_vs_110;
    extern std::string    shiny_vs_130;
    extern std::string    shiny_fs_110;
    extern std::string    shiny_fs_130;
    extern std::string    cayley_fetch;
    extern std::string    metaballs_fetch;
}

using std::cerr;
using std::endl;


class MetaBallField : public glhpmc::Field
{
public:
    struct ProgramContext : public glhpmc::Field::ProgramContext
    {
        GLint   m_loc_centers;
        GLint   m_loc_twist;
    };

    MetaBallField( glhpmc::HPMCConstants* constants,
                   unsigned int samples_x,
                   unsigned int samples_y,
                   unsigned int samples_z )
        : glhpmc::Field( constants, samples_x, samples_y, samples_z )
    {
        animate( 0.f);
    }

    Field::ProgramContext*
    createContext( GLuint program ) const
    {
        ProgramContext* c = new ProgramContext;
        c->m_loc_centers = glGetUniformLocation( program, "centers" );
        c->m_loc_twist   = glGetUniformLocation( program, "twist" );
        return c;
    }


    bool
    gradients() const
    {
        return false;
    }

    bool
    bind( Field::ProgramContext* program_context ) const
    {
        if( ProgramContext* c = dynamic_cast<ProgramContext*>( program_context ) ) {
            glUniform1f( c->m_loc_twist, m_twist );
            glUniform3fv( c->m_loc_centers, 8, m_centers );
        }
        return true;
    }


    const std::string
    fetcherSource(bool gradient) const
    {
        return resources::metaballs_fetch;
    }

    void
    animate( GLfloat t )
    {
        for(int i=0; i<8; i++) {
            m_centers[3*i+0] = 0.5+0.3*sin( t+sin(0.1*t)*i );
            m_centers[3*i+1] = 0.5+0.3*cos( 0.9*t+sin(0.1*t)*i );
            m_centers[3*i+2] = 0.5+0.3*cos( 0.7*t+sin(0.01*t)*i );
        }
        m_twist = 5.0*sin(0.1*t);
    }


private:
    GLfloat     m_centers[ 3*8 ];
    GLfloat     m_twist;

};

int                             volume_size_x       = 64;
int                             volume_size_y       = 64;
int                             volume_size_z       = 64;
float                           iso                 = 10.f;
GLuint                          shiny_p             = 0;
GLint                           shiny_loc_pm        = -1;
GLint                           shiny_loc_nm        = -1;
GLuint                          flat_p              = 0;
GLint                           flat_loc_pm         = -1;
GLint                           flat_loc_color      = -1;
glhpmc::HPMCConstants*           hpmc_c              = NULL;
glhpmc::HPMCIsoSurface*          hpmc_h              = NULL;
glhpmc::HPMCIsoSurfaceRenderer*  hpmc_th_shiny       = NULL;
glhpmc::HPMCIsoSurfaceRenderer*  hpmc_th_flat        = NULL;
MetaBallField*                  hpmc_field          = NULL;


// -----------------------------------------------------------------------------
// a metaball evaluation shader, with domain twist

void
printHelp( const std::string& appname )
{
    cerr << "HPMC demo application that visalizes a set of meta balls with domain twist"<<endl<<endl;
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


// -----------------------------------------------------------------------------
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
    hpmc_field = new MetaBallField( hpmc_c, volume_size_x, volume_size_y, volume_size_z );
    hpmc_h =  glhpmc::HPMCIsoSurface::factory( hpmc_c, hpmc_field );


    // --- shiny shaded render pipeline ----------------------------------------
    {
        hpmc_th_shiny = HPMCcreateIsoSurfaceRenderer( hpmc_h );
        char* traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shiny );

        const GLchar* vs_src[2] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::shiny_vs_110.c_str()
            : resources::shiny_vs_130.c_str(),
            traversal_code
        };
        const GLchar* fs_src[1] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::shiny_fs_110.c_str()
            : resources::shiny_fs_130.c_str()
        };

        GLuint vs = glCreateShader( GL_VERTEX_SHADER );
        glShaderSource( vs, 2, vs_src, NULL );
        compileShader( vs, "shiny vertex shader" );

        GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
        glShaderSource( fs, 1, fs_src, NULL );
        compileShader( fs, "shiny fragment shader" );

        shiny_p = glCreateProgram();
        glAttachShader( shiny_p, vs );
        glAttachShader( shiny_p, fs );
        linkProgram( shiny_p, "shiny program" );
        shiny_loc_pm = glGetUniformLocation( shiny_p, "PM" );
        shiny_loc_nm = glGetUniformLocation( shiny_p, "NM" );

        HPMCsetIsoSurfaceRendererProgram( hpmc_th_shiny, shiny_p, 0, 1, 2 );

        glDeleteShader( vs );
        glDeleteShader( fs );
        free( traversal_code );
    }
    // --- flat-shaded render pipeline -----------------------------------------
    {
        hpmc_th_flat = HPMCcreateIsoSurfaceRenderer( hpmc_h );
        char* traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shiny );

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

// -----------------------------------------------------------------------------
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
    hpmc_field->animate( t );


    // Clear screen
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Build histopyramid
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
        glUseProgram( shiny_p );
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 1.0-iso, 0.0, iso );
        }
        else {
            glUniformMatrix4fv( shiny_loc_pm, 1, GL_FALSE, PM );
            glUniformMatrix3fv( shiny_loc_nm, 1, GL_FALSE, NM );
        }
        HPMCextractVertices( hpmc_th_shiny, GL_FALSE );
    }
    else {
        // Wireframe rendering
        glUseProgram( flat_p );
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
            glColor3f( 0.2*(1.0-iso), 0.0, 0.2*iso );
        }
        else {
            glUniformMatrix4fv( flat_loc_pm, 1, GL_FALSE, PM );
            glUniform4f( flat_loc_color,  0.5f, 0.5f, 1.f, 1.f );
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
