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

#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iterator>
#include <GL/glew.h>


#if defined(__unix) || defined(__APPLE__)
#include <sys/time.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>
#endif



/*
#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/glut.h>
#endif
*/
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include "common.hpp"

using std::min;
using std::max;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::copy;
using std::back_insert_iterator;

double              aspect_x=1.0;
double              aspect_y=1.0;
bool                wireframe = false;
bool                record = false;
bool                is_binary = false;
glhpmc::HPMCTarget          hpmc_target     = glhpmc::HPMC_TARGET_GL20_GLSL110;
glhpmc::HPMCDebugBehaviour  hpmc_debug      = glhpmc::HPMC_DEBUG_NONE;


// === misc matrix operations ==================================================

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
#define ASSERT_GL do {                                                         \
    GLenum err = glGetError();                                                 \
    if( err != GL_NO_ERROR ) {                                                 \
        cerr << __FILE__ << '@' << __LINE__ << ": OpenGL error:"               \
             << err << endl;                                                   \
        /*exit( EXIT_FAILURE );*/                                                  \
    }                                                                          \
} while(0);

// -----------------------------------------------------------------------------
void
checkFramebufferStatus( const std::string& file, const int line )
{
    GLenum status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
    if( status != GL_FRAMEBUFFER_COMPLETE_EXT ) {
        cerr << __FILE__ << '@' << __LINE__ << ": Framebuffer error: "
             << status << endl;
        exit( EXIT_FAILURE );
    }
}

// -----------------------------------------------------------------------------
double
getTimeOfDay()
{
#if defined(__unix) || defined(__APPLE__)
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec+tv.tv_usec*1e-6;
#elif defined(_WIN32)
    LARGE_INTEGER f;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return t.QuadPart/(double) f.QuadPart;
#else
    return 0;
#endif
}

// --- compile shader and check for errors -------------------------------------
void
compileShader( GLuint shader, const string& what )
{
    glCompileShader( shader );

    GLint compile_status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &compile_status );
    if( compile_status != GL_TRUE ) {
        cerr << "Compilation of " << what << " failed, infolog:" << endl;

        GLint logsize;
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            vector<GLchar> infolog( logsize+1 );
            glGetShaderInfoLog( shader, logsize, NULL, &infolog[0] );
            cerr << string( infolog.begin(), infolog.end() ) << endl;
        }
        else {
            cerr << "Empty log message" << endl;
        }
        cerr << "Exiting." << endl;
        exit( EXIT_FAILURE );
    }
}

// --- compile program and check for errors ------------------------------------
void
linkProgram( GLuint program, const string& what )
{
    glLinkProgram( program );

    GLint linkstatus;
    glGetProgramiv( program, GL_LINK_STATUS, &linkstatus );
    if( linkstatus != GL_TRUE ) {
        cerr << "Linking of " << what << " failed, infolog:" << endl;

        GLint logsize;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            vector<GLchar> infolog( logsize+1 );
            glGetProgramInfoLog( program, logsize, NULL, &infolog[0] );
            cerr << string( infolog.begin(), infolog.end() ) << endl;
        }
        else {
            cerr << "Empty log message" << endl;
        }
        cerr << "Exiting." << endl;
        exit( EXIT_FAILURE );
    }
}

// --- set a list of varyings as active ----------------------------------------
void
activateVaryings( GLuint program,
                  GLsizei count,
                  const GLchar** names )
{
    for(GLsizei i=0; i<count; i++) {
        glActiveVaryingNV( program, names[i] );
    }
}

// --- enable a list of varyings for feedback transform ------------------------
void
setFeedbackVaryings( GLuint program,
                     GLsizei count,
                     const GLchar** names )
{
    vector<GLint> locs( count );
    for( GLsizei i=0; i<count; i++ ) {
        locs[i] = glGetVaryingLocationNV( program, names[i] );
        if( locs[i] == -1 ) {
            cerr << "Failed to get varying location of "
                 << names[i] << "\n";
            exit( EXIT_FAILURE );
        }
    }
    glTransformFeedbackVaryingsNV( program,
                                   count,
                                   &locs[0],
                                   GL_INTERLEAVED_ATTRIBS_NV );
}

// -----------------------------------------------------------------------------
void
keyboard( unsigned char key, int x, int y )
{
    if(key == 'r') {
        record = !record;
    }
    else if(key == 'w') {
        wireframe = !wireframe;
    }
    else if(key == 'q' || key == 27) {
        exit( EXIT_SUCCESS );
    }
}

// -----------------------------------------------------------------------------
void
reshape(int w, int h)
{
    if( w > h ) {
        aspect_x = (double)w/(double)h;
        aspect_y = 1.0;
    } else {
        aspect_x = 1.0;
        aspect_y = (double)h/(double)w;
    }
    if( w > 0 && h > 0 ) {
        glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    }
    glutPostRedisplay();
}

// -----------------------------------------------------------------------------
void
idle()
{
        glutPostRedisplay();
}


// --- calculate fps and call render loop func ---------------------------------

#ifdef SINTEF_INTERNAL
struct frame_info
{
    frame_info( float t, float fps, bool wf )
            : m_t(t), m_fps( fps ), m_wf(wf) {}
    float m_t;
    float m_fps;
    bool  m_wf;
};
#include "/work/projects/siut/siut/io_utils/DumpFrames.hpp"
#endif

void
display()
{
    if( (hpmc_debug != glhpmc::HPMC_DEBUG_KHR_DEBUG) && (hpmc_debug != glhpmc::HPMC_DEBUG_KHR_DEBUG_VERBOSE) ) {
        GLenum error = glGetError();
        while( error != GL_NO_ERROR ) {
            std::cerr << "Render loop entered with GL error " << std::hex << error << std::endl;
            error = glGetError();
        }
    }

    double t = getTimeOfDay();;
    static double pt;
    static double start;
    static double last_fps_t;
    static int frames = 0;
    static double fps = 0.0;
    static bool first = true;
    if( first ) {
        first = false;
        start = t;
        last_fps_t = pt = t-start;
    }
    t = t-start;
    float dt = max(0.0, min(1.0, t-pt));
    frames++;
    if( t-last_fps_t > (1.0/30.0 ) ) {
        fps = frames / (t-last_fps_t);
        last_fps_t = t;
        frames = 0;
    }

    GLfloat P[16];
    GLfloat MV[16];
    GLfloat PMV[16];
    GLfloat NM[9];
    GLfloat T[16];

    GLfloat MVi[16];

    frustum( P,  -0.2*aspect_x, 0.2*aspect_x, -0.2*aspect_y, 0.2*aspect_y, 0.5, 3.0 );

    translate( MV, 0.f, 0.f, -2.f );
    rotX( T, 20.f );
    rightMulAssign( MV, T );
    rotY( T, 20.f*t );
    rightMulAssign( MV, T );
    translate( T, -0.5f, -0.5f, -0.5f );
    rightMulAssign( MV, T );
    extractUpperLeft3x3( NM, MV );

    translate( MVi, 0.5f, 0.5f, 0.5f );
    rotY( T, -20.f*t );
    rightMulAssign( MVi, T );
    rotX( T, -20.f );
    rightMulAssign( MVi, T );
    translate( T, 0.f, 0.f, 2.f );
    rightMulAssign( MVi, T );


    memcpy( PMV, P, sizeof(GLfloat)*16 );
    rightMulAssign( PMV, MV );


#ifdef SINTEF_INTERNAL
    static std::vector<frame_info> sequence;
    static int seq_p = 0;
    static float rpt;
    static float rlf;
    if( record ) {
        sequence.push_back( frame_info( t, fps, wireframe ) );
        float rpt = sequence[0].m_t;
        float rlf = sequence[0].m_t;
        render( t, dt, fps );
    }
    else if( seq_p < sequence.size() ) {
        float rt = sequence[ seq_p ].m_t;
        float rfps = sequence[ seq_p ].m_fps;
        wireframe = sequence[ seq_p ].m_wf;
        float rdt = rt - rpt;
        render( rt, rdt, rfps );
        if( (rdt==0.0) || ((rt-rlf) > (1.0/60.0)) ) {
            int no_frames = floorf( (rt-rlf)*60.0 );
            std::cerr << "storing " << no_frames << " frame(s) at t=" << t << ", dt = " << dt << "\n";
            for(int i=0; i<no_frames; i++) {
                siut::io_utils::dumpFrames( "/work/frame_" );
            }
            rlf = rt - ((rt-rlf)*60.0-floorf( (rt-rlf)*60.0 ))/60.0;
        }
        rpt = rt;
        seq_p++;
    }
    else if (seq_p > 0) {
        keyboard( 'q', 0, 0 );
    }
    else {
        render( t, dt, fps );
    }
#else
    render( t, dt, fps, P, MV, PMV, NM, MVi );
#endif

    static std::string message = "";

    if( floor(5.0*(t-dt)) != floor(5.0*(t)) ) {
        message = infoString( fps );
        if( glhpmc::HPMC_TARGET_GL31_GLSL140 < hpmc_target ) {
            std::cerr << message << std::endl;
        }
    }
    if( hpmc_target <= glhpmc::HPMC_TARGET_GL31_GLSL140 ) {
        glUseProgram( 0 );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
        glDisable( GL_DEPTH_TEST );
        glColor3f( 1.0, 1.0, 1.0 );
        glRasterPos2f( -0.99, 0.95 );
        for(int i=0; i<message.size(); i++) {
            glutBitmapCharacter( GLUT_BITMAP_8_BY_13, (int)message[i] );
        }
    }

    pt = t;
    glutSwapBuffers();

    if( (hpmc_debug != glhpmc::HPMC_DEBUG_KHR_DEBUG) && (hpmc_debug != glhpmc::HPMC_DEBUG_KHR_DEBUG_VERBOSE) ) {
        GLenum error = glGetError();
        while( error != GL_NO_ERROR ) {
            std::cerr << "Render loop produced GL error " << std::hex << error << std::endl;
            error = glGetError();
        }
    }
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
    std::cerr << "OpenGL debug [src=";
    switch( source ) {
    case GL_DEBUG_SOURCE_API:               std::cerr << "api"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:     std::cerr << "wsy"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:   std::cerr << "cmp"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:       std::cerr << "3py"; break;
    case GL_DEBUG_SOURCE_APPLICATION:       std::cerr << "app"; break;
    case GL_DEBUG_SOURCE_OTHER:             std::cerr << "oth"; break;
    default:                                std::cerr << "???"; break;
    }

    std::cerr << ", type=";
    switch( type ) {
    case GL_DEBUG_TYPE_ERROR:               std::cerr << "error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cerr <<  "deprecated"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cerr <<  "undef"; break;
    case GL_DEBUG_TYPE_PORTABILITY:         std::cerr <<  "portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE:         std::cerr <<  "performance"; break;
    case GL_DEBUG_TYPE_OTHER:               std::cerr <<  "other"; break;
    default:                                std::cerr << "???"; break;
    }

    std::cerr << ", severity=";
    switch( severity ) {
    case GL_DEBUG_SEVERITY_HIGH:            std::cerr <<  "high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM:          std::cerr <<  "medium"; break;
    case GL_DEBUG_SEVERITY_LOW:             std::cerr <<  "low"; break;
    default:                                std::cerr << "???"; break;
    }

    std::cerr << "] " << message << std::endl;
}


void
printOptions()
{
    std::cerr << "Options that control profile:" << std::endl;
    std::cerr << "    --target-gl20  OpenGL 2.0, GLSL 110" << std::endl;
    std::cerr << "    --target-gl21  OpenGL 2.1, GLSL 120" << std::endl;
    std::cerr << "    --target-gl30  OpenGL 3.0, GLSL 130" << std::endl;
    std::cerr << "    --target-gl31  OpenGL 3.1, GLSL 140" << std::endl;
    std::cerr << "    --target-gl32  OpenGL 3.2, GLSL 150" << std::endl;
    std::cerr << "    --target-gl33  OpenGL 3.3, GLSL 330" << std::endl;
    std::cerr << "    --target-gl40  OpenGL 4.0, GLSL 400" << std::endl;
    std::cerr << "    --target-gl41  OpenGL 4.1, GLSL 410" << std::endl;
    std::cerr << "    --target-gl42  OpenGL 4.2, GLSL 420" << std::endl;
    std::cerr << "    --target-gl43  OpenGL 4.3, GLSL 430" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options that control error checking:" << std::endl;
    std::cerr << "    --debug-none               No GL error checking." << std::endl;
    std::cerr << "    --debug-stderr             Use glGetError to check for errors." << std::endl;
    std::cerr << "    --debug-stderr-verbose     Use glGetError to check for errors." << std::endl;
    std::cerr << "    --debug-khr-debug          Use GL_KHR_debug to check for errors." << std::endl;
    std::cerr << "    --debug-khr-debug-verbose  Use GL_KHR_debug to check for errors." << std::endl;
}


// -----------------------------------------------------------------------------
int
main(int argc, char **argv)
{
    glutInit( &argc, argv );

    bool target_set = false;
    for( int i=1; i<argc; ) {
        int eat = 0;
        std::string arg( argv[1] );
        if( arg == "--target-gl20" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL20_GLSL110;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl21" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL21_GLSL120;
            eat = 1;
            target_set = true;
        }
        else if(  arg == "--target-gl23" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL30_GLSL130;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl31" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL31_GLSL140;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl32" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL32_GLSL150;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl33" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL33_GLSL330;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl40" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL40_GLSL400;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl41" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL41_GLSL410;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl42" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL42_GLSL420;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--target-gl43" ) {
            hpmc_target = glhpmc::HPMC_TARGET_GL43_GLSL430;
            eat = 1;
            target_set = true;
        }
        else if( arg == "--debug-none" ) {
            hpmc_debug = glhpmc::HPMC_DEBUG_NONE;
            eat = 1;
        }
        else if( arg == "--debug-stderr" ) {
            hpmc_debug = glhpmc::HPMC_DEBUG_STDERR;
            eat = 1;
        }
        else if( arg == "--debug-stderr-verbose" ) {
            hpmc_debug = glhpmc::HPMC_DEBUG_STDERR_VERBOSE;
            eat = 1;
        }
        else if( arg == "--debug-khr" ) {
            hpmc_debug = glhpmc::HPMC_DEBUG_KHR_DEBUG;
            eat = 1;
        }
        else if( arg == "--debug-khr-verbose" ) {
            hpmc_debug = glhpmc::HPMC_DEBUG_KHR_DEBUG_VERBOSE;
            eat = 1;
        }
        else if( arg == "--binary" ) {
            is_binary = true;
            eat = 1;
        }
        else if( arg == "--no-binary" ) {
            is_binary = false;
            eat = 1;
        }
        else if(arg == "--help" ) {
            printHelp( argv[0] );
            exit( -1 );
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

    if( target_set ) {
        switch( hpmc_target ) {
        case glhpmc::HPMC_TARGET_GL20_GLSL110:
            glutInitContextVersion( 2, 0 );
            break;
        case glhpmc::HPMC_TARGET_GL21_GLSL120:
            glutInitContextVersion( 2, 1 );
            break;
        case glhpmc::HPMC_TARGET_GL30_GLSL130:
            glutInitContextVersion( 3, 0 );
            break;
        case glhpmc::HPMC_TARGET_GL31_GLSL140:
            glutInitContextVersion( 3, 1 );
            break;
        case glhpmc::HPMC_TARGET_GL32_GLSL150:
            glutInitContextVersion( 3, 2 );
            break;
        case glhpmc::HPMC_TARGET_GL33_GLSL330:
            glutInitContextVersion( 3, 3 );
            break;
        case glhpmc::HPMC_TARGET_GL40_GLSL400:
            glutInitContextVersion( 4, 0 );
            break;
        case glhpmc::HPMC_TARGET_GL41_GLSL410:
            glutInitContextVersion( 4, 1 );
            break;
        case glhpmc::HPMC_TARGET_GL42_GLSL420:
            glutInitContextVersion( 4, 2 );
            break;
        case glhpmc::HPMC_TARGET_GL43_GLSL430:
            glutInitContextVersion( 4, 3 );
            break;
        }
    }

    if( (hpmc_debug == glhpmc::HPMC_DEBUG_KHR_DEBUG) || (hpmc_debug == glhpmc::HPMC_DEBUG_KHR_DEBUG_VERBOSE ) ) {
        glutInitContextFlags( GLUT_CORE_PROFILE | GLUT_DEBUG );
    }
    else {
        glutInitContextFlags( GLUT_CORE_PROFILE );
    }

    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutInitWindowSize( 1280, 720 );
    glutCreateWindow( argv[0] );
    GLenum error = glGetError();
    while( error != GL_NO_ERROR ) {
        std::cerr << "Context creation created GL error " << std::hex << error << std::endl;
        error = glGetError();
    }
    glewExperimental = GL_TRUE;
    GLenum glew_error = glewInit();
    if( glew_error != GLEW_OK ) {
        std::cerr << "GLEW failed to initialize." << std::endl;
        exit( EXIT_FAILURE );
    }
    error = glGetError();
    while( error != GL_NO_ERROR ) {
        std::cerr << "GLEW initialization created GL error " << std::hex << error << std::endl;
        error = glGetError();
    }

    GLint gl_major, gl_minor;
    glGetIntegerv( GL_MAJOR_VERSION, &gl_major );
    glGetIntegerv( GL_MINOR_VERSION, &gl_minor );
    std::cerr << "Driver reports OpenGL " << gl_major << "." << gl_minor << "." << std::endl;
    if( !target_set ) {
        // target not set, use driver version as target
        if( gl_major < 2 ) {
            std::cerr << "Requires minimum OpenGL 2.0.";
            exit( EXIT_FAILURE );
        }
        else if( gl_major == 2 ) {
            if( gl_minor == 0 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL20_GLSL110;
            }
            else {
                hpmc_target = glhpmc::HPMC_TARGET_GL21_GLSL120;
            }
        }
        else if( gl_major == 3 ) {
            if( gl_minor == 0 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL30_GLSL130;
            }
            else if (gl_minor == 1 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL31_GLSL140;
            }
            else if (gl_minor == 2 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL32_GLSL150;
            }
            else {
                hpmc_target = glhpmc::HPMC_TARGET_GL33_GLSL330;
            }
        }
        else if( gl_major == 4 ) {
            if( gl_minor == 0 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL40_GLSL400;
            }
            else if (gl_minor == 1 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL41_GLSL410;
            }
            else if (gl_minor == 2 ) {
                hpmc_target = glhpmc::HPMC_TARGET_GL42_GLSL420;
            }
            else {
                hpmc_target = glhpmc::HPMC_TARGET_GL43_GLSL430;
            }
        }
        else {
            hpmc_target = glhpmc::HPMC_TARGET_GL43_GLSL430;
        }
    }

    switch( hpmc_target ) {
    case glhpmc::HPMC_TARGET_GL20_GLSL110:
        std::cerr << "HPMC target is OpenGL 2.0" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL21_GLSL120:
        std::cerr << "HPMC target is OpenGL 2.1" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL30_GLSL130:
        std::cerr << "HPMC target is OpenGL 3.0" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL31_GLSL140:
        std::cerr << "HPMC target is OpenGL 3.1" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL32_GLSL150:
        std::cerr << "HPMC target is OpenGL 3.2" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL33_GLSL330:
        std::cerr << "HPMC target is OpenGL 3.3" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL40_GLSL400:
        std::cerr << "HPMC target is OpenGL 4.0" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL41_GLSL410:
        std::cerr << "HPMC target is OpenGL 4.1" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL42_GLSL420:
        std::cerr << "HPMC target is OpenGL 4.2" << std::endl;
        break;
    case glhpmc::HPMC_TARGET_GL43_GLSL430:
        std::cerr << "HPMC target is OpenGL 4.3" << std::endl;
        break;
    }


    if( (hpmc_debug == glhpmc::HPMC_DEBUG_KHR_DEBUG) || (hpmc_debug == glhpmc::HPMC_DEBUG_KHR_DEBUG_VERBOSE) ) {
        if( glewIsSupported( "GL_KHR_debug" ) ) {
            glEnable( GL_DEBUG_OUTPUT_SYNCHRONOUS );
            glDebugMessageCallback( debugLogger, NULL );
            glDebugMessageControl( GL_DONT_CARE,
                                   GL_DONT_CARE,
                                   hpmc_debug == glhpmc::HPMC_DEBUG_KHR_DEBUG_VERBOSE ? GL_DEBUG_SEVERITY_LOW : GL_DEBUG_SEVERITY_MEDIUM,
                                   0, NULL, GL_TRUE );
        }
        else {
            std::cerr << "GL_KHR_debug extension not present, reverting to stderr.\n";
            hpmc_debug = glhpmc::HPMC_DEBUG_STDERR;
        }
    }
    error = glGetError();
    while( error != GL_NO_ERROR ) {
        std::cerr << "Debug setup created GL error " << std::hex << error << std::endl;
        error = glGetError();
    }

    glutReshapeFunc( reshape );
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    init( argc, argv );
    glutMainLoop();
    return EXIT_SUCCESS;
}


