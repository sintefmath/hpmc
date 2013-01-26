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
#include <glhpmc/Constants.hpp>
#include <glhpmc/VertexCountTable.hpp>
#include <glhpmc/Logger.hpp>

using namespace HPMC;
static const std::string package = "HPMC.VertexCountTable";

HPMCVertexCountTable::HPMCVertexCountTable( HPMCConstants*  constants )
    : m_constants( constants ),
      m_texture( 0 )
{

}

void
HPMCVertexCountTable::init(  )
{
    Logger log( m_constants, package + ".init" );

    std::vector<GLfloat> table( 256 );
    for(size_t j=0; j<256; j++) {
        size_t count;
        for(count=0; count<16; count++) {
            if( HPMC_triangle_table[j][count] == -1 ) {
                break;
            }
        }
        table[ remapCode(j) ] = static_cast<GLfloat>( count );
    }

    glActiveTexture( GL_TEXTURE0 );
    glGenTextures( 1, &m_texture );
    log.setObjectLabel( GL_TEXTURE, m_texture, "vertex count table" );

    glBindTexture( GL_TEXTURE_1D, m_texture );
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130) {
        glTexImage1D( GL_TEXTURE_1D, 0,
                      GL_ALPHA32F_ARB, 256, 0,
                      GL_ALPHA, GL_FLOAT,
                      table.data() );
    }
    else {
        glTexImage1D( GL_TEXTURE_1D, 0,
                      GL_R32F, 256, 0,
                      GL_RED, GL_FLOAT,
                      table.data() );
    }
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_1D, 0 );
}


HPMCVertexCountTable::~HPMCVertexCountTable()
{
    if( m_texture != 0 ) {
        glDeleteTextures( 1, &m_texture );
    }
}
