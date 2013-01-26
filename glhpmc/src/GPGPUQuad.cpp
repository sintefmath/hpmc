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
#include <vector>
#include <glhpmc/glhpmc_internal.hpp>
#include <glhpmc/GPGPUQuad.hpp>
#include <glhpmc/Constants.hpp>
#include <glhpmc/Logger.hpp>

namespace glhpmc {

static const std::string package = "HPMC.GPGPUQuad";

HPMCGPGPUQuad::HPMCGPGPUQuad(  HPMCConstants* constants  )
    : m_constants( constants ),
      m_vbo( 0 ),
      m_vao( 0 ),
      m_pass_through_vs( 0 )
{ }

void
HPMCGPGPUQuad::init()
{
    Logger log( m_constants, package + ".init" );

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, sizeof(GLfloat)*3*4, HPMC_gpgpu_quad_vertices, GL_STATIC_DRAW );

    if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
        glGenVertexArrays( 1, &m_vao );
        glBindVertexArray( m_vao );
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );
        glEnableVertexAttribArray( 0 );
        glBindVertexArray( 0 );
    }

    // generate pass-through vertex shader
    std::string src =
            m_constants->versionString() +
            "// generated by HPMC.GPGPUQuad.init\n";
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        src +=  "void\n"
                "main()\n"
                "{\n"
                "    gl_TexCoord[0] = 0.5*gl_Vertex+vec4(0.5);\n"
                "    gl_Position    = gl_Vertex;\n"
                "}\n";
    }
    else {
        src +=  "in vec3 vertex;\n"
                "out vec2 texcoord;\n"
                "void\n"
                "main()\n"
                "{\n"
                "    texcoord    = 0.5*vertex.xy + vec2(0.5);\n"
                "    gl_Position = vec4( vertex, 1.0);\n"
                "}\n";
    }
    m_pass_through_vs = HPMCcompileShader( src, GL_VERTEX_SHADER );

}

HPMCGPGPUQuad::~HPMCGPGPUQuad()
{
    Logger log( m_constants, package + ".destructor" );

    if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 && (m_vao != 0) ) {
        glDeleteVertexArrays( 1, &m_vao );
    }
    if( m_vbo != 0 ) {
        glDeleteBuffers( 1, &m_vbo );
    }
    if( m_pass_through_vs != 0 ) {
        glDeleteShader( m_pass_through_vs );
    }
}

void
HPMCGPGPUQuad::bindVertexInputs() const
{
    Logger log( m_constants, package + ".bindVertexInputs" );
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        glVertexPointer( 3, GL_FLOAT, 0, NULL );
        glEnableClientState( GL_VERTEX_ARRAY );
    }
    else {
        glBindVertexArray( m_vao );
    }
}

bool
HPMCGPGPUQuad::configurePassThroughVertexShader( GLuint program ) const
{
    Logger log( m_constants, package + ".configurePassThroughVertexShader" );
    if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
        glBindAttribLocation( program, 0, "vertex" );
    }
    return true;
}

void
HPMCGPGPUQuad::render() const
{
    glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
}

} // of namespace glhpmc
