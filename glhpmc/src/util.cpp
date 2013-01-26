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

namespace glhpmc {

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



} // of namespace glhpmc
