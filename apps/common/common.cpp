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
#define snprintf _snprintf_s
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
bool                binary = false;
HPMCTarget          hpmc_target     = HPMC_TARGET_GL20_GLSL110;
HPMCDebugBehaviour  hpmc_debug      = HPMC_DEBUG_NONE;


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
    GLenum error = glGetError();
    while( error != GL_NO_ERROR ) {
        error = glGetError();
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

    frustum( P,  -0.2*aspect_x, 0.2*aspect_x, -0.2*aspect_y, 0.2*aspect_y, 0.5, 3.0 );
    translate( MV, 0.f, 0.f, -2.f );
    rotX( T, 20.f );
    rightMulAssign( MV, T );
    rotY( T, 20.f*t );
    rightMulAssign( MV, T );
    translate( T, -0.5f, -0.5f, -0.5f );
    rightMulAssign( MV, T );
    extractUpperLeft3x3( NM, MV );

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
    render( t, dt, fps, P, MV, PMV, NM );
#endif

    static std::string message = "";

    if( floor(5.0*(t-dt)) != floor(5.0*(t)) ) {
        message = infoString( fps );
        if( HPMC_TARGET_GL31_GLSL140 < hpmc_target ) {
            std::cerr << message << std::endl;
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
        for(int i=0; i<message.size(); i++) {
            glutBitmapCharacter( GLUT_BITMAP_8_BY_13, (int)message[i] );
        }
    }

    pt = t;
    glutSwapBuffers();

    error = glGetError();
    while( error != GL_NO_ERROR ) {
        fprintf( stderr, "render loop produced GL error %x\n", error );
        error = glGetError();
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

    GLint gl_major, gl_minor;
    glGetIntegerv( GL_MAJOR_VERSION, &gl_major );
    glGetIntegerv( GL_MINOR_VERSION, &gl_minor );
    if( gl_major < 2 ) {
        std::cerr << "Requires minimum OpenGL 2.0 (driver reports "
                  << gl_major << "."
                  << gl_minor << ").";
        exit( EXIT_FAILURE );
    }
    else if( gl_major == 2 ) {
        if( gl_minor == 0 ) {
            hpmc_target = HPMC_TARGET_GL20_GLSL110;
        }
        else {
            hpmc_target = HPMC_TARGET_GL21_GLSL120;
        }
    }
    else if( gl_major == 3 ) {
        if( gl_minor == 0 ) {
            hpmc_target = HPMC_TARGET_GL30_GLSL130;
        }
        else if( gl_minor == 1 ) {
            hpmc_target = HPMC_TARGET_GL31_GLSL140;
        }
        else if( gl_minor == 2 ) {
            hpmc_target = HPMC_TARGET_GL32_GLSL150;
        }
        else {
            hpmc_target = HPMC_TARGET_GL33_GLSL330;
        }
    }
    else if( gl_major == 4 ) {
        if( gl_minor == 0 ) {
            hpmc_target = HPMC_TARGET_GL40_GLSL400;
        }
        else if( gl_minor == 1 ) {
            hpmc_target = HPMC_TARGET_GL41_GLSL410;
        }
        else if( gl_minor == 2 ) {
            hpmc_target = HPMC_TARGET_GL42_GLSL420;
        }
        else {
            hpmc_target = HPMC_TARGET_GL43_GLSL430;
        }
    }
    else {
        hpmc_target = HPMC_TARGET_GL43_GLSL430;
    }


    for( int i=1; i<argc; ) {
        int eat = 0;
        if( strcmp( argv[i], "--target-gl20" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL20_GLSL110;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl21" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL21_GLSL120;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl30" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL30_GLSL130;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl31" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL31_GLSL140;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl32" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL32_GLSL150;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl33" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL33_GLSL330;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl40" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL40_GLSL400;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl41" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL41_GLSL410;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl42" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL42_GLSL420;
            eat = 1;
        }
        else if( strcmp( argv[i], "--target-gl43" ) == 0 ) {
            hpmc_target = HPMC_TARGET_GL43_GLSL430;
            eat = 1;
        }
        else if( strcmp( argv[i], "--debug-none" ) == 0 ) {
            hpmc_debug = HPMC_DEBUG_NONE;
            eat = 1;
        }
        else if( strcmp( argv[i], "--debug-stderr" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_STDERR;
            eat = 1;
        }
        else if( strcmp( argv[i], "--debug-stderr-verbose" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_STDERR_VERBOSE;
            eat = 1;
        }
        else if( strcmp( argv[i], "--debug-khr-debug" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_KHR_DEBUG;
            eat = 1;
        }
        else if( strcmp( argv[i], "--debug-khr-debug-verbose" ) == 0  ) {
            hpmc_debug = HPMC_DEBUG_KHR_DEBUG_VERBOSE;
            eat = 1;
        }
        else if( strcmp( argv[i], "--help" ) == 0 ) {
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

    switch( hpmc_target ) {
    case HPMC_TARGET_GL20_GLSL110:
        glutInitContextVersion( 2, 0 );
        std::cerr << "Target is OpenGL 2.0" << std::endl;
        break;
    case HPMC_TARGET_GL21_GLSL120:
        glutInitContextVersion( 2, 1 );
        std::cerr << "Target is OpenGL 2.1" << std::endl;
        break;
    case HPMC_TARGET_GL30_GLSL130:
        glutInitContextVersion( 3, 0 );
        std::cerr << "Target is OpenGL 3.0" << std::endl;
        break;
    case HPMC_TARGET_GL31_GLSL140:
        glutInitContextVersion( 3, 1 );
        std::cerr << "Target is OpenGL 3.1" << std::endl;
        break;
    case HPMC_TARGET_GL32_GLSL150:
        glutInitContextVersion( 3, 2 );
        std::cerr << "Target is OpenGL 3.2" << std::endl;
        break;
    case HPMC_TARGET_GL33_GLSL330:
        glutInitContextVersion( 3, 3 );
        std::cerr << "Target is OpenGL 3.3" << std::endl;
        break;
    case HPMC_TARGET_GL40_GLSL400:
        glutInitContextVersion( 4, 0 );
        std::cerr << "Target is OpenGL 4.0" << std::endl;
        break;
    case HPMC_TARGET_GL41_GLSL410:
        glutInitContextVersion( 4, 1 );
        std::cerr << "Target is OpenGL 4.1" << std::endl;
        break;
    case HPMC_TARGET_GL42_GLSL420:
        glutInitContextVersion( 4, 2 );
        std::cerr << "Target is OpenGL 4.2" << std::endl;
        break;
    case HPMC_TARGET_GL43_GLSL430:
        glutInitContextVersion( 4, 3 );
        std::cerr << "Target is OpenGL 4.3" << std::endl;
        break;
    }

    if( (hpmc_debug == HPMC_DEBUG_KHR_DEBUG) || (hpmc_debug == HPMC_DEBUG_KHR_DEBUG_VERBOSE ) ) {
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
    init( argc, argv );
    glutMainLoop();
    return EXIT_SUCCESS;
}


