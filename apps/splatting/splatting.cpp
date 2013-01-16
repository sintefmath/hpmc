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

// Extracting iso-surfaces from a scalar field stored on disc.
//
// This example demonstrates the most basic use of HPMC, providing the scalar
// field as a 3D texture. The example gets volume dimensions and a file name of
// a 8-bit raw dataset from the command line, reads the data into a 3D texture
// and passes this texture to HPMC.
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
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <GL/glew.h>
#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/glut.h>
#endif
#include "hpmc.h"
#include "../common/common.hpp"

using std::ifstream;
using std::vector;
using std::string;
using std::cerr;
using std::endl;

vector<GLubyte>                 dataset;
int                             volume_size_x  = 128;
int                             volume_size_y  = 128;
int                             volume_size_z  = 128;
float                           iso            = 10.f;
GLuint                          volume_tex     = 0;
GLuint                          splat_fbo      = 0;
GLuint                          splat_v        = 0;
GLuint                          splat_f        = 0;
GLuint                          splat_p        = 0;
GLuint                          particle_v     = 0;
GLuint                          particle_p     = 0;
GLuint                          shaded_v       = 0;
GLuint                          shaded_f       = 0;
GLuint                          shaded_p       = 0;
GLuint                          flat_v         = 0;
GLuint                          flat_p         = 0;
struct HPMCConstants*           hpmc_c         = NULL;
struct HPMCIsoSurface*          hpmc_h         = NULL;
struct HPMCIsoSurfaceRenderer*  hpmc_th_flat   = NULL;
struct HPMCIsoSurfaceRenderer*  hpmc_th_shaded = NULL;

GLuint                          splat_vbo      = 0;
GLuint                          positions_p    = 0;
GLuint                          gauss_tex      = 0;
GLsizei                         positions_N    = 5024;
GLuint                          positions_tex[2];
GLuint                          positions_vbo[2];

namespace resources{
	extern std::string splatting_vs_140; 
	extern std::string splatting_fs_140;
    extern std::string particle_vs_140;
    extern std::string shiny_vs_130;
    extern std::string shiny_fs_130;
    extern std::string solid_vs_130;
}
// -----------------------------------------------------------------------------


        


GLfloat splat_geo[12] =
{
    -1.0f, -1.0f, 0.0,
     1.0f, -1.0f, 0.0,
     1.0f,  1.0f, 0.0,
    -1.0f,  1.0f, 0.0,
};


// -----------------------------------------------------------------------------
void
init()
{
    // --- create texture with Gauss function ----------------------------------

    size_t GN = 1024;
    std::vector<GLfloat> buf( GN );
    for(size_t i=0; i<GN; i++) {
        float sigma = 0.28;
        float x = (float)i/GN;
        float g = (1.0/sqrt(2.0*M_PI*sigma))*exp(-x*x/(2.0*sigma*sigma));
        buf[i] = g/4.0;
    }
    glGenTextures( 1, &gauss_tex );
    glBindTexture( GL_TEXTURE_1D, gauss_tex );
    glTexImage1D( GL_TEXTURE_1D, 0, GL_R32F, GN, 0, GL_RED, GL_FLOAT, &buf[0] );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glBindTexture( GL_TEXTURE_1D, 0 );

    // --- create empty volume ------------------------------------------------------

    GLint alignment;
    glGetIntegerv( GL_UNPACK_ALIGNMENT, &alignment );
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &volume_tex );
    glBindTexture( GL_TEXTURE_3D, volume_tex );
    glTexImage3D( GL_TEXTURE_3D, 0, GL_RGBA16F,
                  volume_size_x, volume_size_y, volume_size_z, 0,
                  GL_RED, GL_FLOAT, NULL );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_3D, 0 );
    glPixelStorei( GL_UNPACK_ALIGNMENT, alignment );

    glGenFramebuffers( 1, &splat_fbo );

    glGenBuffers( 1, &splat_vbo );


    glBindBuffer( GL_ARRAY_BUFFER, splat_vbo );
    glBufferData( GL_ARRAY_BUFFER, 12*sizeof( GLfloat ), &splat_geo[0], GL_STATIC_DRAW );

    glGenBuffers( 2, &positions_vbo[0] );
    positions_p = 0;
    glBindBuffer( GL_ARRAY_BUFFER, positions_vbo[0] );
    glBufferData( GL_ARRAY_BUFFER, positions_N*8*sizeof( GLfloat ), NULL, GL_DYNAMIC_COPY );
    GLfloat* ptr = reinterpret_cast<GLfloat*>( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
    srand( 42 );
    for(int i=0; i<positions_N; i++) {
        *ptr++ = 0.0f + cos( 0.001*i );
        *ptr++ = 0.0f + sin( 0.001*i );
        *ptr++ = -0.1f;
        *ptr++ = 0.0f;
        *ptr++ = 0.5f// + 0.4*cos( 0.002*i )
                 + 0.01*static_cast<GLfloat>( rand() )/ static_cast<GLfloat>( RAND_MAX );
        *ptr++ = 0.5f// + 0.4*sin( 0.002*i )
                 + 0.01*static_cast<GLfloat>( rand() )/ static_cast<GLfloat>( RAND_MAX );
        *ptr++ = 0.9f;
        *ptr++ = 1.0f;
    }
    glUnmapBuffer( GL_ARRAY_BUFFER );
    glBindBuffer( GL_ARRAY_BUFFER, positions_vbo[1] );
    glBufferData( GL_ARRAY_BUFFER, positions_N*8*sizeof( GLfloat ), NULL, GL_DYNAMIC_COPY );

    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    glGenTextures( 2, &positions_tex[0] );
    glBindTexture( GL_TEXTURE_BUFFER, positions_tex[0] );
    glTexBuffer( GL_TEXTURE_BUFFER, GL_RGBA32F, positions_vbo[ 0 ] );
    glBindTexture( GL_TEXTURE_BUFFER, positions_tex[1] );
    glTexBuffer( GL_TEXTURE_BUFFER, GL_RGBA32F, positions_vbo[ 1 ] );
    glBindTexture( GL_TEXTURE_BUFFER, positions_tex[0] );

    const char* splat_v_src_pa[1] =
    {
        resources::splatting_vs_140.c_str()
    };

    splat_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( splat_v, 1, &splat_v_src_pa[0], NULL );
    compileShader( splat_v, "splat vertex shader" );

    const char* splat_f_src_pa[1] =
    {
        resources::splatting_fs_140.c_str()
    };

    splat_f = glCreateShader( GL_FRAGMENT_SHADER );
    glShaderSource( splat_f, 1, &splat_f_src_pa[0], NULL );
    compileShader( splat_f, "splat fragment shader" );

    // link program
    splat_p = glCreateProgram();
    glAttachShader( splat_p, splat_v );
    glAttachShader( splat_p, splat_f );
    linkProgram( splat_p, "splat program" );
    glUseProgram( splat_p );
    glUniform1i( glGetUniformLocation( splat_p, "positions" ), 0 );
    glUniform1i( glGetUniformLocation( splat_p, "gauss" ), 1 );
    glUseProgram( 0 );
    //ASSERT_GL;


    const char* particle_v_src_pa[1] =
    {
        resources::particle_vs_140.c_str()
    };

    particle_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( particle_v, 1, &particle_v_src_pa[0], NULL );
    compileShader( particle_v, "particle vertex shader" );

    const char* particle_varying_names[2] =
    {
        "vel",
        "pos"
    };

    // link program
    particle_p = glCreateProgram();
    glAttachShader( particle_p, particle_v );
//     for(GLuint i=0; i<2; i++) {
//         glActiveVaryingNV( particle_p, particle_varying_names[i] );
//     }
    linkProgram( particle_p, "particle program" );
//     GLint particle_varying_locs[2];
//     for(GLuint i=0; i<2; i++) {
//         particle_varying_locs[i] = glGetVaryingLocation( particle_p,
//                                                            particle_varying_names[i] );
//         std::cerr << particle_varying_locs[i] << "\n";
//     }
	//need to use varying locations instead of text 
    glTransformFeedbackVaryings( particle_p,
                                   2, &particle_varying_names[0],
                                   GL_INTERLEAVED_ATTRIBS );
    glUseProgram( particle_p );
    glUniform1i( glGetUniformLocation( particle_p, "density" ), 0 );
    glUseProgram( 0 );
   // ASSERT_GL;


    // --- create HistoPyramid -------------------------------------------------
    hpmc_c = HPMCcreateConstants( hpmc_target, hpmc_debug );
    hpmc_h = HPMCcreateIsoSurface( hpmc_c );

    HPMCsetLatticeSize( hpmc_h,
                        volume_size_x,
                        volume_size_y,
                        volume_size_z );

    HPMCsetGridSize( hpmc_h,
                     volume_size_x-1,
                     volume_size_y-1,
                     volume_size_z-1 );

    float max_size = std::max( volume_size_x, std::max( volume_size_y, volume_size_z ) );
    HPMCsetGridExtent( hpmc_h,
                       volume_size_x / max_size,
                       volume_size_y / max_size,
                       volume_size_z / max_size );

    HPMCsetFieldTexture3D( hpmc_h,
                           volume_tex,
						   GL_ALPHA,
						   GL_RGB );


    // --- create traversal vertex shader --------------------------------------
    hpmc_th_shaded = HPMCcreateIsoSurfaceRenderer( hpmc_h );

    char *traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shaded );
    const char* shaded_vsrc[2] =
    {
        resources::shiny_vs_130.c_str(),
		traversal_code

    };
    shaded_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( shaded_v, 2, &shaded_vsrc[0], NULL );
    compileShader( shaded_v, "shaded vertex shader" );
    free( traversal_code );

    const char* shaded_fsrc[1] =
    {
        resources::shiny_fs_130.c_str()
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
        resources::solid_vs_130.c_str(),
		traversal_code
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
    HPMCsetIsoSurfaceRendererProgram( hpmc_th_flat,
                                      flat_p,
                                      0, 1, 2 );


    glPolygonOffset( 1.0, 1.0 );
    wireframe = true;
	glBindFramebuffer( GL_FRAMEBUFFER, splat_fbo );
	GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
	if (status != GL_FRAMEBUFFER_COMPLETE )
	{
		cerr << "Error occurred when checking FBO completeness!\n" << status <<endl;

	}
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

}

// -----------------------------------------------------------------------------
void
render( float t, float dt, float fps, const GLfloat* P, const GLfloat* MV, const GLfloat* PMV, const GLfloat *NM )
{
   // --- clear screen and set up view ----------------------------------------
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    //    glMatrixMode( GL_PROJECTION );
    //    glLoadIdentity();
    //    glFrustum( -0.2*aspect_x, 0.2*aspect_x, -0.2*aspect_y, 0.2*aspect_y, 0.5, 3.0 );

    //    glMatrixMode( GL_MODELVIEW );
    //    glLoadIdentity();
    //    glTranslatef( 0.0f, 0.0f, -2.0f );
    //    glRotatef( 13.*t, 0.0, 0.0, 1.0 );
    ////    glRotatef( 20.0*t, 1.0, 0.0, 0.0 );
    //    glRotatef( -90, 1.0, 0.0, 0.0 );

    float max_size = std::max( volume_size_x, std::max( volume_size_y, volume_size_z ) );
    //    glTranslatef( -0.5f*volume_size_x / max_size,
    //                  -0.5f*volume_size_y / max_size,
    //                  -0.5f*volume_size_z / max_size );


//    ASSERT_GL;

    GLsizei viewport[4];
    glGetIntegerv( GL_VIEWPORT, &viewport[0] );
    glPushClientAttribDefaultEXT( GL_CLIENT_ALL_ATTRIB_BITS );

    glBindFramebuffer( GL_FRAMEBUFFER, splat_fbo );
    glViewport( 0, 0, volume_size_x, volume_size_y );

    glUseProgram( splat_p );
    glUniform1i( glGetUniformLocation( splat_p, "active" ), static_cast<GLint>( std::min( 10000.0f, 500*t ) ) );

    GLint slice_loc = glGetUniformLocation( splat_p, "slice_z" );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_BUFFER, positions_tex[positions_p] );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_1D, gauss_tex );

    glBindBuffer( GL_ARRAY_BUFFER, splat_vbo );
    glInterleavedArrays( GL_V3F, 0, NULL );
    glClearColor( 0.0, 0.0, 0.0, 0.0 );
    glEnable( GL_BLEND );
    glBlendFunc( GL_ONE, GL_ONE );
    for(int i=0; i<volume_size_z; i++) {
        glFramebufferTexture3D( GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_3D,
                                   volume_tex,
                                   0,
                                   i );
       // checkFramebufferStatus( __FILE__, __LINE__ );

        glClear( GL_COLOR_BUFFER_BIT );

        glUniform1f( slice_loc, (GLfloat)i/volume_size_z );
        glDrawArraysInstanced( GL_QUADS, 0, 4, positions_N );
    }

	GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
	if (status != GL_FRAMEBUFFER_COMPLETE )
	{
		cerr << "Error occurred when checking FBO completeness!\n" << status <<endl;

	}

    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_1D, 0 );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_BUFFER, 0 );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glDisable( GL_BLEND );
    glViewport( viewport[0], viewport[1], viewport[2], viewport[3] );
    glPopClientAttrib();


    //    glPushClientAttribDefaultEXT( GL_CLIENT_ALL_ATTRIB_BITS );
    glUseProgram( particle_p );
    glBindTexture( GL_TEXTURE_3D, volume_tex );

    glUniform1f( glGetUniformLocation( particle_p, "dt" ), dt );
    glUniform1i( glGetUniformLocation( particle_p, "active" ), static_cast<GLint>( std::min( 10000.0f, 500*t ) ) );

    glUniform3f( glGetUniformLocation( particle_p, "gravity" ),
                 -1.f*sinf(13.0*(M_PI/180.f)*t), 0.f, -1.f*cosf( 13.0*(M_PI/180.f)*t) );
//                 cosf( 1.3*t), 0.f, sinf(1.3*t) );


    glColor3f( 1, 1, 1 );

    glBindBuffer( GL_ARRAY_BUFFER, positions_vbo[ positions_p ] );
    glInterleavedArrays( GL_T4F_V4F, 0, NULL );
    positions_p = (positions_p+1)%2;
    glBindBufferBase( GL_TRANSFORM_FEEDBACK_BUFFER,
                        0, positions_vbo[ positions_p ] );


    if( !wireframe) {
        glEnable( GL_RASTERIZER_DISCARD_NV );
    }
    else {
        glPointSize( 10 );
        glEnable( GL_POINT_SMOOTH );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE );
        glEnable( GL_BLEND );
        glDepthMask( GL_FALSE );
    }
    //probably wrong, check with working transform feedback
    glBeginTransformFeedback( GL_POINTS );
    glDrawArrays( GL_POINTS, 0, positions_N );
    glEndTransformFeedback( );
    if( !wireframe) {
        glDisable( GL_RASTERIZER_DISCARD_NV );
    }
    else {
        glDisable( GL_POINT_SMOOTH );
        glDisable( GL_BLEND );
        glDepthMask( GL_TRUE );
    }

    glBindTexture( GL_TEXTURE_3D, 0 );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glPopClientAttrib();

    // --- build HistoPyramid --------------------------------------------------
    iso = .3;//.5 + 0.48*cosf( t );
    HPMCbuildIsoSurface( hpmc_h, iso );

    // --- render surface ------------------------------------------------------

    glEnable( GL_DEPTH_TEST );
    if( !wireframe ) {
        glColor3f( 0.8, 0.8, 1.0 );
        HPMCextractVertices( hpmc_th_shaded, GL_FALSE );
    }
    else {
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE );
        glDepthMask( GL_FALSE );
        glColor4f( 0.3, 0.3, 0.5, 0.15 );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable( GL_POLYGON_OFFSET_FILL );
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glDisable( GL_POLYGON_OFFSET_FILL );


        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glColor4f( 0.5, 0.5, 0.8, 0.2 );
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable( GL_BLEND );
        glDepthMask( GL_TRUE );
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
		<< " triangles, iso=" << iso
		<< (wireframe?"[wireframe]":"");
	return o.str();
}

void printHelp( const std::string& appname )
{
	cerr << "HPMC demo application extracting iso-surfaces from a scalar field stored on disc.\n"<<endl;
	cerr << "Usage: " << appname << " [options] xsize [ysize zsize] \n" <<endl;
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
init(int argc, char **argv)
 {
	 if( hpmc_target < HPMC_TARGET_GL30_GLSL130){
		 cerr << "HPMC Splatting demo requires at least OpenGL 3.0 to run, exiting." << endl;
		 exit( EXIT_FAILURE );
	 }

     if( argc == 2 ) {
         volume_size_x = volume_size_y = volume_size_z = atoi( argv[1] );
     }
     else if( argc == 4 ) {
         volume_size_x = atoi( argv[1] );
         volume_size_y = atoi( argv[2] );
         volume_size_z = atoi( argv[3] );
     }
    init();
}
