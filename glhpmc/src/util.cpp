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
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <glhpmc/glhpmc.hpp>
#include <glhpmc/glhpmc_internal.hpp>

using std::string;
using std::vector;
using std::stringstream;
using std::setw;
using std::cerr;
using std::endl;


// -----------------------------------------------------------------------------
std::string
HPMCaddLineNumbers( const std::string& src )
{
    stringstream out;

    int line = 1;
    for( string::const_iterator it = src.begin(); it!=src.end(); ++it ) {
        string::const_iterator jt = it;
        int c=0;
        out << std::setw(3) << line << ": ";
        for(; *jt != '\n' && jt != src.end(); jt++) {
            out << *jt;
            c++;
        }
        out << "\n";
        line ++;
        it = jt;
        if(jt == src.end() )
            break;
    }

    return out.str();
}

/*
static void
HPMClogShaderSource( HPMC::OldLogger log, const std::string& src )
{
    if( src.empty() ) {
        return;
    }
    std::stringstream o;
    int line = 1;
    for( string::const_iterator it = src.begin(); it!=src.end(); ) {
        o.str("");
        string::const_iterator jt = it;
        o << "src" << std::setw(3) << line << ": ";
        for(; (*jt != '\0') && (*jt != '\n') && (*jt != '\r') && (jt != src.end()); jt++) {
            o << *jt;
        }
        for(; (*jt == '\0' || *jt == '\n' || *jt=='\r') && jt != src.end(); jt++ ) {}
        HPMCLOG_DEBUG( log, o.str() );
        line ++;
        if( it != jt ) {
            it = jt;
        }
        else {
            it++;
        }
        if(it == src.end() )
            break;
    }
}

static void
HPMClogErrorMessage( HPMC::OldLogger log, const std::string& msg )
{
    if( msg.empty() ) {
        return;
    }
    std::stringstream o;
    for( string::const_iterator it = msg.begin(); it!=msg.end(); it ) {
        o.str("");
        string::const_iterator jt = it;
        for(; *jt != '\0' && *jt != '\n' && *jt != '\r' && jt != msg.end(); jt++) {
            o << *jt;
        }
        for(; (*jt == '\0' || *jt == '\n' || *jt=='\r') && jt != msg.end(); jt++ ) {}
        HPMCLOG_ERROR( log, "err " << o.str() );
        if( it != jt ) {
            it = jt;
        }
        else {
            it++;
        }
        if(it == msg.end() )
            break;
    }
}

GLuint
HPMCcompileShader( HPMC::OldLogger log, const std::string& src, GLuint type )
{
    GLuint shader = glCreateShader( type );

    // glShaderSource wants an array of string pointers
    const char* p = src.c_str();
    glShaderSource( shader, 1, &p, NULL );
    glCompileShader( shader );

    // check if everything is ok
    GLint status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
    if( status == GL_TRUE ) {
        // successful compilation
        return shader;
    }
    std::string error_msg;
    GLint logsize;
    glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );
    if( logsize > 0 ) {
        vector<GLchar> infolog( logsize+1 );
        glGetShaderInfoLog( shader, logsize, NULL, &infolog[0] );
        error_msg = string( infolog.begin(), infolog.end() );
    }
    // compilation failed
    HPMCLOG_ERROR( log, "Compilation of shader failed " );
    HPMClogShaderSource( log, src );
    HPMClogErrorMessage( log, error_msg );
    glDeleteShader( shader );
    return 0u;
}
*/

// -----------------------------------------------------------------------------
GLuint
HPMCcompileShader( const std::string& src, GLuint type )
{
    GLuint shader = glCreateShader( type );

    // glShaderSource wants an array of string pointers
    const char* p = src.c_str();
    glShaderSource( shader, 1, &p, NULL );
    glCompileShader( shader );

    // check if everything is ok
    GLint status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
    if( status == GL_TRUE ) {
        // successful compilation
        return shader;
    }

    // compilation failed
#ifdef DEBUG
    cerr << "HPMC error: compilation of shader failed." << endl;
    cerr << "HPMC error: *** shader source code ***" << endl;
    cerr << HPMCaddLineNumbers( src );
    cerr << "HPMC error: *** shader build log ***" << endl;

    // get size of build log
    GLint logsize;
    glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );

    // get build log
    if( logsize == 0 ) {
        cerr << "HPMC error: empty log." << endl;
    }
    else {
        vector<GLchar> infolog( logsize+1 );
        glGetShaderInfoLog( shader, logsize, NULL, &infolog[0] );
        cerr << string( infolog.begin(), infolog.end() ) << endl;
    }
#endif

    glDeleteShader( shader );
    return 0u;
}

// -----------------------------------------------------------------------------
bool
HPMClinkProgram( GLuint program )
{
    glLinkProgram( program );

    GLint linkstatus;
    glGetProgramiv( program, GL_LINK_STATUS, &linkstatus );
    if( linkstatus == GL_TRUE ) {
        // successful link
        return true;
    }

    // linking failed
#ifdef DEBUG
    cerr << "HPMC: linking of program failed." << endl;
    cerr << "HPMC: *** program link log ***" << endl;

    // get size of build log
    GLint logsize;
    glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );

    // get build log
    if( logsize == 0 ) {
        cerr << "HPMC: empty log." << endl;
    }
    else {
        vector<GLchar> infolog( logsize+1 );
        glGetProgramInfoLog( program, logsize, NULL, &infolog[0] );
        cerr << string( infolog.begin(), infolog.end() ) << endl;
    }
#endif

    return false;
}

// -----------------------------------------------------------------------------
GLint
HPMCgetUniformLocation( GLuint program, const std::string& name )
{
    GLint loc = glGetUniformLocation( program, name.c_str() );
#ifdef DEBUG
    if( loc < 0 ) {
        cerr << "HPMC warning: failed to locate uniform \"" << name << "\"." << endl;
    }
#endif
    return loc;
}

// -----------------------------------------------------------------------------
/*
void
HPMCrenderGPGPUQuad( struct HPMCHistoPyramid* h )
{
    if( h->m_constants->m_target < HPMC_TARGET_GL30_GLSL130 ) {
        glBindBuffer( GL_ARRAY_BUFFER, h->m_constants->m_gpgpu_quad_vbo );
        glVertexPointer( 3, GL_FLOAT, 0, NULL );
        glEnableClientState( GL_VERTEX_ARRAY );
    }
    else {
        glBindVertexArray( h->m_constants->m_gpgpu_quad_vao );
    }
    glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
}
*/
