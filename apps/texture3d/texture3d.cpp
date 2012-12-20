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
#include <iostream>
#include <fstream>
#include <vector>
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

int volume_size_x;
int volume_size_y;
int volume_size_z;
vector<GLubyte> dataset;

GLuint volume_tex;

struct HPMCConstants* hpmc_c;
struct HPMCIsoSurface* hpmc_h;

GLuint shaded_v;
GLuint shaded_f;
GLuint shaded_p;
struct HPMCIsoSurfaceRenderer* hpmc_th_flat;

GLuint flat_v;
GLuint flat_p;
struct HPMCIsoSurfaceRenderer* hpmc_th_shaded;

/*
int
main(int argc, char **argv)
{
    glutInit( &argc, argv );

    if( argc != 5 ) {
        cerr << "HPMC demo application that visualizes raw volumes."<<endl<<endl;
        cerr << "Usage: " << argv[0] << " xsize ysize zsize rawfile"<<endl<<endl;
        cerr << "where: xsize    The number of samples in the x-direction."<<endl;
        cerr << "       ysize    The number of samples in the y-direction."<<endl;
        cerr << "       zsize    The number of samples in the z-direction."<<endl;
        cerr << "       rawfile  Filename of a raw set of bytes describing"<<endl;
        cerr << "                the volume."<<endl<<endl;
        cerr << "Raw volumes can be found e.g. at http://www.volvis.org."<<endl<<endl;
        cerr << "Example usage:"<<endl;
        cerr << "    " << argv[0] << " 64 64 64 neghip.raw"<< endl;
        cerr << "    " << argv[0] << " 256 256 256 foot.raw"<< endl;
        cerr << "    " << argv[0] << " 256 256 178 BostonTeapot.raw"<< endl;
        cerr << "    " << argv[0] << " 301 324 56 lobster.raw"<< endl;
        exit( EXIT_FAILURE );
    }
    else {
        volume_size_x = atoi( argv[1] );
        volume_size_y = atoi( argv[2] );
        volume_size_z = atoi( argv[3] );

        dataset.resize( volume_size_x * volume_size_y * volume_size_z );
        ifstream datafile( argv[4], std::ios::in | std::ios::binary );
        if( datafile.good() ) {
            datafile.read( reinterpret_cast<char*>( &dataset[0] ),
                           dataset.size() );
        }
        else {
            cerr << "Error opening \"" << argv[4] << "\" for reading." << endl;
            exit( EXIT_FAILURE );
        }
    }
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutInitWindowSize( 2598, 3307 );
//    glutInitWindowSize( 1280, 720 );
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
*/

// -----------------------------------------------------------------------------
void
init()
{
    // --- upload volume ------------------------------------------------------

    GLint alignment;
    glGetIntegerv( GL_UNPACK_ALIGNMENT, &alignment );
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &volume_tex );
    glBindTexture( GL_TEXTURE_3D, volume_tex );
    glTexImage3D( GL_TEXTURE_3D, 0, GL_ALPHA,
                  volume_size_x, volume_size_y, volume_size_z, 0,
                  GL_ALPHA, GL_UNSIGNED_BYTE, &dataset[0] );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_3D, 0 );
    glPixelStorei( GL_UNPACK_ALIGNMENT, alignment );

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

    float max_size = std::max( volume_size_x, std::max( volume_size_y, volume_size_z ) );
    HPMCsetGridExtent( hpmc_h,
                       volume_size_x / max_size,
                       volume_size_y / max_size,
                       volume_size_z / max_size );

    HPMCsetFieldTexture3D( hpmc_h,
                           volume_tex,
                           GL_FALSE );

    // --- create traversal vertex shader --------------------------------------
    hpmc_th_shaded = HPMCcreateIsoSurfaceRenderer( hpmc_h );

    char *traversal_code = HPMCisoSurfaceRendererShaderSource( hpmc_th_shaded );
    const char* shaded_vsrc[2] =
    {
        traversal_code,
        shaded_vertex_shader_110.c_str()
    };
    shaded_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( shaded_v, 2, &shaded_vsrc[0], NULL );
    compileShader( shaded_v, "shaded vertex shader" );
    free( traversal_code );

    const char* shaded_fsrc[1] =
    {
        shaded_fragment_shader_110.c_str()
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
        flat_vertex_shader_110.c_str()
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
}

// -----------------------------------------------------------------------------
void
render( float t, float dt, float fps )
{
    // --- clear screen and set up view ----------------------------------------
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -0.15*aspect_x, 0.15*aspect_x, -0.15*aspect_y, 0.15*aspect_y, 0.5, 3.0 );

//    t = 5.5;

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0.0f, 0.0f, -2.0f );
    glRotatef( 20.0*t, 0.0, 1.0, 0.0 );
    glRotatef( -90, 1.0, 0.0, 0.0 );



    float max_size = std::max( volume_size_x, std::max( volume_size_y, volume_size_z ) );
    glTranslatef( -0.5f*volume_size_x / max_size,
                  -0.5f*volume_size_y / max_size,
                  -0.5f*volume_size_z / max_size );

    // --- build HistoPyramid --------------------------------------------------
    float iso = 0.5 + 0.48*cosf( t );
//    float iso = 0.21;//0.5 + 0.48*cosf( t );
    HPMCbuildIsoSurface( hpmc_h, iso );

    // --- render surface ------------------------------------------------------
    glEnable( GL_DEPTH_TEST );
    if( !wireframe ) {
        glColor3f( 1.0-iso, 0.0, iso );
        HPMCextractVertices( hpmc_th_shaded, GL_FALSE );
    }
    else {
        glColor3f( 0.2*(1.0-iso), 0.0, 0.2*iso );
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable( GL_POLYGON_OFFSET_FILL );
        HPMCextractVertices( hpmc_th_flat, GL_FALSE );
        glDisable( GL_POLYGON_OFFSET_FILL );

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
        glColor3f( 1.0, 1.0, 1.0 );
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
//        glutBitmapCharacter( GLUT_BITMAP_8_BY_13, (int)message[i] );
    }
}

