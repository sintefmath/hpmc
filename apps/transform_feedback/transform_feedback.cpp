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

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include "../common/common.hpp"

using std::cerr;
using std::endl;
using std::vector;
using std::string;


enum TransformFeedbackExtensions
{
    USE_EXT,
    USE_NV,
    USE_CORE
}                               extension           = USE_CORE;
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
GLuint                          plain_p              = 0;
GLint                           plain_loc_pm         = -1;
GLint                           plain_loc_color      = -1;
GLuint                          mc_tri_vbo          = 0;
GLsizei                         mc_tri_vbo_N        = 0;
GLuint                          vao                 = 0;
struct glhpmc::HPMCConstants*           hpmc_c              = NULL;
struct glhpmc::HPMCIsoSurface*          hpmc_h              = NULL;
struct glhpmc::HPMCIsoSurfaceRenderer*  hpmc_th_shaded      = NULL;
struct glhpmc::HPMCIsoSurfaceRenderer*  hpmc_th_flat        = NULL;

namespace resources {
    extern std::string plain_vs_110;
    extern std::string plain_vs_130;
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
    cerr << "Options specific for this app:" << std::endl;
    cerr << "    --ext   Use EXT_transform_feedback extension" << std::endl;
    cerr << "    --nv    Use NV_transform_feedback extension" << std::endl;
    cerr << "    --core  Use 3.0 and up core transform feedback" << std::endl;
    cerr << endl;
    printOptions();
}

void
init( int argc, char** argv )
{
    // handle command line options specific for this app
    for( int i=1; i<argc; ) {
        int eat = 0;
        std::string arg( argv[i] );
        if( arg == "--ext" ) {
            extension = USE_EXT;
            eat = 1;
        }
        else if( arg == "--nv" ) {
            extension = USE_NV;
            eat = 1;
        }
        else if( arg == "--core" ) {
            extension = USE_CORE;
            eat = 1;
        }
        if( eat ) {
            argc = argc - eat;
            for( int k=i; k<argc; k++ ) {
                argv[k] = argv[k+eat];
            }
        }
        else {
            i++;
        }
    }

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
    if( volume_size_x < 4 ) {
        cerr << "Volume size x < 4" << endl;
        exit( EXIT_FAILURE );
    }
    if( volume_size_y < 4 ) {
        cerr << "Volume size y < 4" << endl;
        exit( EXIT_FAILURE );
    }
    if( volume_size_z < 4 ) {
        cerr << "Volume size z < 4" << endl;
        exit( EXIT_FAILURE );
    }

    switch( extension ) {
    case USE_EXT:
        if( !GLEW_EXT_transform_feedback ) {
            cerr << "Missing EXT_transform_feedback." << endl;
            exit( EXIT_FAILURE );
        }
        break;
    case USE_NV:
        if( !GLEW_NV_transform_feedback ) {
            cerr << "Missing NV_transform_feedback." << endl;
            exit( EXIT_FAILURE );
        }
        break;
    case USE_CORE:
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
            cerr << "Insufficient target for core transform feedback." << endl;
            exit( EXIT_FAILURE );
        }
        break;
    }


    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = glhpmc::HPMCConstants::factory( hpmc_target, hpmc_debug );
    hpmc_h = glhpmc::HPMCIsoSurface::factory( hpmc_c );

    hpmc_h->setLatticeSize( volume_size_x, volume_size_y, volume_size_z );
    hpmc_h->setGridSize( volume_size_x-1, volume_size_y-1, volume_size_z-1 );
    hpmc_h->setGridExtent( 1.f, 1.f, 1.f );

    glhpmc::HPMCsetFieldCustom( hpmc_h,
                        resources::cayley_fetch.c_str(),
                        0,
                        GL_TRUE );


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
    // --- flat-shaded render pipeline with transform feedback capture ---------
    {
        hpmc_th_flat = glhpmc::HPMCcreateIsoSurfaceRenderer( hpmc_h );
        char* traversal_code = glhpmc::HPMCisoSurfaceRendererShaderSource( hpmc_th_flat );

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
        const char* varyings[2] = {
            "position"
        };

        GLuint vs = glCreateShader( GL_VERTEX_SHADER );
        glShaderSource( vs, 2, vs_src, NULL );
        compileShader( vs, "flat vertex shader" );

        GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
        glShaderSource( fs, 1, fs_src, NULL );
        compileShader( fs, "flat fragment shader" );

        flat_p = glCreateProgram();
        glAttachShader( flat_p, vs );
        glAttachShader( flat_p, fs );
        if( glhpmc::HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
            glBindFragDataLocation( flat_p, 0, "fragment" );
        }
        // When using the EXT extension (or 3.0 core), we can directly tag varyings
        // for feedback by name before linkage. The NV extension requires that we
        // first tag the varyings as active, link the program, determine the
        // varying locations of the varyings that shall be fed back, and then ship
        // this to GL.
        switch( extension ) {
        case USE_EXT:
            glTransformFeedbackVaryingsEXT( flat_p, 1, varyings, GL_INTERLEAVED_ATTRIBS_EXT );
            break;
        case USE_NV:
            // tag the varyings we will record as active (so they don't get
            // optimized away).
            for(int i=0; i<1; i++ ) {
                glActiveVaryingNV( flat_p, varyings[i] );
            }
            break;
        case USE_CORE:
            glTransformFeedbackVaryings( flat_p, 1, varyings, GL_INTERLEAVED_ATTRIBS );
            break;
        }
        linkProgram( flat_p, "flat program" );
        if( extension == USE_NV ) {
            GLint varying_locs[1];
            for(int i=0; i<1; i++) {
                varying_locs[i] = glGetVaryingLocationNV( flat_p, varyings[i] );
            }
            glTransformFeedbackVaryingsNV( flat_p, 1, varying_locs,  GL_INTERLEAVED_ATTRIBS );
        }

        flat_loc_pm = glGetUniformLocation( flat_p, "PM" );
        flat_loc_color = glGetUniformLocation( flat_p, "color" );

        // Associate program with traversal handle
        HPMCsetIsoSurfaceRendererProgram( hpmc_th_flat, flat_p, 0, 1, 2 );

        glDeleteShader( vs );
        glDeleteShader( fs );
        free( traversal_code );
    }
    // --- plain rendering of a triangle set in a VBO --------------------------
    {
        const GLchar* vs_src[1] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::plain_vs_110.c_str()
            : resources::plain_vs_130.c_str()
        };
        const GLchar* fs_src[1] = {
            hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130
            ? resources::solid_fs_110.c_str()
            : resources::solid_fs_130.c_str()
        };

        GLuint vs = glCreateShader( GL_VERTEX_SHADER );
        glShaderSource( vs, 1, vs_src, NULL );
        compileShader( vs, "plain vertex shader" );

        GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
        glShaderSource( fs, 1, fs_src, NULL );
        compileShader( vs, "plain fragment shader" );

        plain_p = glCreateProgram();
        glAttachShader( plain_p, vs );
        glAttachShader( plain_p, fs );
        if( glhpmc::HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
            glBindAttribLocation( plain_p, 0, "position" );
            glBindFragDataLocation( plain_p, 0, "fragment" );
        }
        linkProgram( plain_p, "plain program" );
        plain_loc_pm = glGetUniformLocation( plain_p, "PM" );
        plain_loc_color = glGetUniformLocation( plain_p, "color" );

        glDeleteShader( vs );
        glDeleteShader( fs );
    }


    glPolygonOffset( 1.0, 1.0 );

    // --- set up buffer for feedback of MC triangles --------------------------
    glGenBuffers( 1, &mc_tri_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, mc_tri_vbo );
    mc_tri_vbo_N = 3*1000;
    glBufferData( GL_ARRAY_BUFFER,
                  3*mc_tri_vbo_N * sizeof(GLfloat),
                  NULL,
                  GL_DYNAMIC_COPY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // If >= 3.0, set up vertex attrib array
    if( glhpmc::HPMC_TARGET_GL30_GLSL130 <= hpmc_target ) {
        glGenVertexArrays( 1, &vao );
        glBindVertexArray( vao );

        glBindBuffer( GL_ARRAY_BUFFER, mc_tri_vbo );
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );
        glEnableVertexAttribArray(0);
        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        glBindVertexArray( 0 );
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
    // --- build HistoPyramid --------------------------------------------------
    float iso = sin(t);
    HPMCbuildIsoSurface( hpmc_h, iso );

    // --- clear screen and set up view ----------------------------------------
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable( GL_DEPTH_TEST );
    if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf( P );
        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf( MV );
    }

    // --- render solid surface ------------------------------------------------

    // if wireframe, do transform feedback capture
    if(!wireframe) {
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
        // resize buffer if needed
        GLsizei N = HPMCacquireNumberOfVertices( hpmc_h );
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
        switch( extension ) {
        case USE_EXT:
            glBindBufferBaseEXT( GL_TRANSFORM_FEEDBACK_BUFFER_EXT, 0, mc_tri_vbo );
            HPMCextractVerticesTransformFeedbackEXT( hpmc_th_flat, GL_FALSE );
            //glFlush(); // on ATI catalyst 9.10, this is needed to avoid some artefacts
            break;
        case USE_NV:
            glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, mc_tri_vbo );
            HPMCextractVerticesTransformFeedbackNV( hpmc_th_flat, GL_FALSE );
            break;
        case USE_CORE:
            glBindBufferBase( GL_TRANSFORM_FEEDBACK_BUFFER, 0, mc_tri_vbo );
            HPMCextractVerticesTransformFeedback( hpmc_th_flat, GL_FALSE );
            break;
        }
        switch( extension ) {
        case USE_EXT:
            glBindBufferBaseEXT( GL_TRANSFORM_FEEDBACK_BUFFER_EXT, 0, 0 );
            break;
        case USE_NV:
            glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
            break;
        case USE_CORE:
            glBindBufferBase( GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0 );
            break;
        }
        glDisable( GL_POLYGON_OFFSET_FILL );

        // --- render wireframe ------------------------------------------------
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glUseProgram( plain_p );
        if( hpmc_target < glhpmc::HPMC_TARGET_GL30_GLSL130 ) {
            glPushClientAttrib( GL_CLIENT_VERTEX_ARRAY_BIT );
            glColor3f( 1.0, 1.0, 1.0 );
            glBindBuffer( GL_ARRAY_BUFFER, mc_tri_vbo );
            glInterleavedArrays( GL_V3F, 0, NULL );
            glDrawArrays( GL_TRIANGLES, 0, N );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
            glPopClientAttrib();
        }
        else {
            glUniform4f( plain_loc_color, 1.f, 1.f, 1.f, 1.f );
            glUniformMatrix4fv( plain_loc_pm, 1, GL_FALSE, PM );
            glBindVertexArray( vao );
            glDrawArrays( GL_TRIANGLES, 0, N );
            glBindVertexArray( 0 );
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL );
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
