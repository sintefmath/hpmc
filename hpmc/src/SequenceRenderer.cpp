/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
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
#include <algorithm>
#include "SequenceRenderer.hpp"
#include "Constants.hpp"
#include "Logger.hpp"

static const std::string package = "HPMC.SequenceRenderer";

HPMCSequenceRenderer::HPMCSequenceRenderer( HPMCConstants* constants )
    : m_constants( constants ),
      m_batch_size( ~0u ),
      m_vbo(0),
      m_vao(0)
{
}

void
HPMCSequenceRenderer::init()
{
    Logger log( m_constants, package + ".init" );
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        m_batch_size = 3000u;
        glGenBuffers( 1, &m_vbo );
        log.setObjectLabel( GL_BUFFER, m_vbo, "sequence enumeration buffer" );

        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        glBufferData( GL_ARRAY_BUFFER,
                      3*sizeof(GLfloat)*m_batch_size,
                      NULL,
                      GL_STATIC_DRAW );
        GLfloat* ptr = reinterpret_cast<GLfloat*>( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
        for( GLsizei i=0; i<m_batch_size; i++ ) {
            *ptr++ = static_cast<GLfloat>( i );
            *ptr++ = 0.0f;
            *ptr++ = 0.0f;
        }
        glUnmapBuffer( GL_ARRAY_BUFFER );
    }
    else {
        m_batch_size = ~0u;
        m_vbo = 0;
        glGenVertexArrays( 1, &m_vao );
    }
}

HPMCSequenceRenderer::~HPMCSequenceRenderer( )
{
    Logger log( m_constants, package + ".destructor" );
    if( m_vbo != 0 ) {
        glDeleteBuffers( 1, &m_vbo );
    }
    if( m_vao != 0 ) {
        glDeleteVertexArrays( 1, &m_vao );
    }
}

void
HPMCSequenceRenderer::bindVertexInputs() const
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

void
HPMCSequenceRenderer::render( GLint offset_loc, GLsizei num ) const
{
    Logger log( m_constants, package + ".render" );
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        for(GLsizei i=0; i<num; i+= m_batch_size ) {
            glUniform1f( offset_loc, static_cast<GLfloat>( i ) );
            glDrawArrays( GL_TRIANGLES, 0, std::min( num-i, m_batch_size ) );
        }
    }
    else {
        glDrawArrays( GL_TRIANGLES, 0, num );
    }
}

void
HPMCSequenceRenderer::unbindVertexInputs() const
{
    Logger log( m_constants, package + ".unbindVertexInputs" );
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        glDisableClientState( GL_VERTEX_ARRAY );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    else {
        glBindVertexArray( 0 );
    }
}
