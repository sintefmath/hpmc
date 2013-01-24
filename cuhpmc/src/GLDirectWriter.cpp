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
#include <GL/glew.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cuhpmc/AbstractField.hpp>
#include <cuhpmc/GLDirectWriter.hpp>
#include <cuhpmc/IsoSurfaceGLInterop.hpp>

namespace cuhpmc {
    namespace resources {
        extern std::string gl_direct_vs_430;
        extern std::string gl_direct_fs_430;
    } // of namespace resources

GLDirectWriter::GLDirectWriter( IsoSurfaceGLInterop* iso_surface )
    : AbstractWriter( iso_surface ),
      m_program( 0 )
{
    std::stringstream defines;
    defines << "#define CUHPMC_LEVELS            " << m_iso_surface->hp5Levels() << "\n";
    defines << "#define CUHPMC_CHUNKS_X          " << m_iso_surface->hp5Chunks().x << "\n";
    defines << "#define CUHPMC_CHUNKS_Y          " << m_iso_surface->hp5Chunks().y << "\n";
    defines << "#define CUHPMC_CHUNKS_Z          " << m_iso_surface->hp5Chunks().z << "\n";
    defines << "#define CUHPMC_FIELD_ROW_PITCH   " << m_field->width() << "\n";
    defines << "#define CUHPMC_FIELD_SLICE_PITCH " << (m_field->width()*m_field->height()) << "\n";

    const std::string vs_src = "#version 430\n" +
                               defines.str() +
                               resources::gl_direct_vs_430;
    const std::string fs_src = resources::gl_direct_fs_430;
    std::stringstream out;

    GLint vs = compileShader( out, vs_src, GL_VERTEX_SHADER );
    if( vs != 0 ) {
        GLint fs = compileShader( out, fs_src, GL_FRAGMENT_SHADER );
        if( fs != 0 ) {
            m_program = glCreateProgram();
            glAttachShader( m_program, vs );
            glAttachShader( m_program, fs );
            if( linkShaderProgram( out, m_program ) ) {

            }
            else {
                glDeleteProgram( m_program );
                m_program = 0;
            }
            glDeleteShader( fs );
        }
        glDeleteShader( vs );
    }
    if( out.tellp() != 0 ) {
        out << "--- vertex source --------------------------------------------------------------\n";
        dumpSource( out, vs_src );
        out << "--- fragment source ------------------------------------------------------------\n";
        dumpSource( out, fs_src );
        std::cerr << out.str() << "\n";

    }
    if( m_program == 0 ) {
        throw std::runtime_error( "Failed to build shader program" );
    }

}

GLDirectWriter::~GLDirectWriter()
{

}

void
GLDirectWriter::render( GLfloat* modelview_projection,
                        GLfloat* normal_matrix,
                        cudaStream_t stream )
{

}

void
GLDirectWriter::dumpSource( std::stringstream& out, const std::string& input )
{
    int line = 1;
    for( std::string::const_iterator it = input.begin(); it!=input.end(); ++it ) {
        std::string::const_iterator jt = it;
        int c=0;
        out << std::setw(3) << line << ": ";
        for(; *jt != '\n' && jt != input.end(); jt++) {
            out << *jt;
            c++;
        }
        out << "\n";
        line ++;
        it = jt;
        if(jt == input.end() )
            break;
    }
}

GLuint
GLDirectWriter::compileShader( std::stringstream& out, const std::string& src, GLenum type ) const
{
    GLuint shader = glCreateShader( type );
    if( type == 0 ) {
        return 0;
    }
    const GLchar* srcs[1] = {
        src.c_str()
    };
    glShaderSource( shader, 1, srcs, NULL );
    glCompileShader( shader );

    GLint logsize;
    glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );
    if( logsize > 1 ) {
        std::vector<GLchar> infolog( logsize+1 );
        glGetShaderInfoLog( shader, logsize, NULL, infolog.data() );
        switch( type ) {
        case GL_VERTEX_SHADER:
            out << "--- GLSL vertex shader compiler ------------------------------------------------\n";
            break;
        case GL_TESS_CONTROL_SHADER:
            out << "--- GLSL tessellation control shader compiler ----------------------------------\n";
            break;
        case GL_TESS_EVALUATION_SHADER:
            out << "--- GLSL tessellation evaluation shader compiler -------------------------------\n";
            break;
        case GL_GEOMETRY_SHADER:
            out << "--- GLSL geometry shader compiler ----------------------------------------------\n";
            break;
        case GL_FRAGMENT_SHADER:
            out << "--- GLSL fragment shader compiler ----------------------------------------------\n";
            break;
        }
        out << std::string( infolog.data() );
    }

    GLint status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
    if( status != GL_TRUE ) {
        glDeleteShader( shader );
        shader = 0;
    }
    return shader;
}

bool
GLDirectWriter::linkShaderProgram( std::stringstream& out, GLuint program ) const
{
    glLinkProgram( program );

    GLint logsize;
    glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );
    if( logsize > 1 ) {
        std::vector<GLchar> infolog( logsize+1 );
        glGetProgramInfoLog( program, logsize, NULL, infolog.data() );
        out << "--- GLSL linker ----------------------------------------------------------------\n";
        out << std::string( infolog.data() );
    }

    GLint status;
    glGetProgramiv( program, GL_LINK_STATUS, &status );
    return status == GL_TRUE;
}




} // of namespace cuhpmc
