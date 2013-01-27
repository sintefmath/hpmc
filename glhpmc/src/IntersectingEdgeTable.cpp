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
#include <vector>
#include <cmath>
#include <glhpmc/glhpmc_internal.hpp>
#include <glhpmc/IntersectingEdgeTable.hpp>
#include <glhpmc/Constants.hpp>
#include <glhpmc/Logger.hpp>

namespace glhpmc {

static const std::string package = "HPMC.IntersectingEdgeTable";

HPMCIntersectingEdgeTable::HPMCIntersectingEdgeTable( HPMCConstants* constants )
    : m_constants( constants ),
      m_tex( 0 ),
      m_tex_flip(0),
      m_tex_normal(0),
      m_tex_normal_flip(0)
{
}

void
HPMCIntersectingEdgeTable::init( )
{
    Logger log( m_constants, package + ".init" );


    std::vector<GLfloat> edge_normals( 256*6*4 );

    // for each marching cubes case
    for(int j=0; j<256; j++) {

        // for each triangle in a case (why 6, should be 5?)
        for(int i=0; i<6; i++) {

            // Pick out the three indices defining a triangle. Snap illegal
            // indices to zero for simplicity (the data will never be used).
            // Then, create pointers to the appropriate vertex positions.
            GLfloat* vp[3];
            for(int k=0; k<3; k++) {
                int ix = std::max( 0, HPMC_triangle_table[j][ std::min(15,3*i+k) ] );
                vp[k] = &( HPMC_midpoint_table[ ix ][0] );
            }

            GLfloat u[3], v[3];
            for(int k=0; k<3; k++) {
                u[k] = vp[2][k]-vp[0][k];
                v[k] = vp[1][k]-vp[0][k];
            }
            GLfloat n[3];
            n[0] = u[1]*v[2] - u[2]*v[1];
            n[1] = u[2]*v[0] - u[0]*v[2];
            n[2] = u[0]*v[1] - u[1]*v[0];
            GLfloat s = 0.5f*std::sqrt( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] );
            for(int k=0; k<3; k++) {
                float tmp = s*n[k]+0.5f;
                if( tmp < 0.f ) tmp  = 0.000001f;
                if( tmp >= 1.f ) tmp = 0.999999f;
                edge_normals[ 4*(6*remapCode(j)+i) + k ] = tmp;
            }
            edge_normals[ 4*(6*remapCode(j)+i) + 3 ] = 0.f;
        }
    }

    std::vector<GLfloat> edge_decode( 256*16*4 );        // shift + axis
    std::vector<GLfloat> edge_decode_normal( 256*16*4 ); // shift + axis + normal vector in fractional part.
    std::vector<GLfloat> edge_decode_flip( 256*16*4 );
    std::vector<GLfloat> edge_decode_normal_flip( 256*16*4 );

    // Build edge decode
    for(int j=0; j<256; j++ ) {
        // Create edges for triangles in one case
        for(int i=0; i<16; i++) {
            int l = HPMC_triangle_table[j][i];
            if( l >= 0 ) {
                for(int k=0; k<4; k++) {
                    edge_decode[4*16*remapCode(j) + 4*i+k ] = HPMC_edge_table[ l ][k];
                }
            }
        }
    }
    // Build edge decode flip (reversed triangles)
    for(int j=0; j<256; j++) {
        for(int i=0; i<5; i++) {
            for(int k=0; k<4; k++) {
                edge_decode_flip[ 4*(16*j+3*i+0)+k ] = edge_decode[ 4*(16*j+3*i+1)+k ];
                edge_decode_flip[ 4*(16*j+3*i+1)+k ] = edge_decode[ 4*(16*j+3*i+0)+k ];
                edge_decode_flip[ 4*(16*j+3*i+2)+k ] = edge_decode[ 4*(16*j+3*i+2)+k ];
            }
        }
    }
    // build edge_decode_normal and edge_decode_normal_flip
    for(int j=0; j<256; j++ ) {
        for(int i=0; i<15; i++) {
            for(int k=0; k<4; k++ ) {


                edge_decode_normal[ 4*(16*j+i) +k ] = edge_decode[ 4*(16*j+i) + k ]
                                                    + edge_normals[ 4*(6*j+i/3 ) + k ];
                edge_decode_normal_flip[ 4*(16*j+i) +k ] = edge_decode_flip[ 4*(16*j+i) + k ]
                                                         + 1 - edge_normals[ 4*(6*j+i/3 ) + k ];
            }
        }
    }

    glActiveTexture( GL_TEXTURE0 );

    glGenTextures( 1, &m_tex );
    log.setObjectLabel( GL_TEXTURE, m_tex, "edge intersect table" );
    glBindTexture( GL_TEXTURE_2D, m_tex );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_RGBA32F_ARB, 16, 256,0,
                  GL_RGBA, GL_FLOAT,
                  edge_decode.data() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    glGenTextures( 1, &m_tex_flip );
    log.setObjectLabel( GL_TEXTURE, m_tex_flip, "flipped edge intersect table" );
    glBindTexture( GL_TEXTURE_2D, m_tex_flip );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_RGBA32F_ARB, 16, 256,0,
                  GL_RGBA, GL_FLOAT,
                  edge_decode_flip.data() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    glGenTextures( 1, &m_tex_normal );
    glBindTexture( GL_TEXTURE_2D, m_tex_normal );
    log.setObjectLabel( GL_TEXTURE, m_tex_normal, "edge intersect normal table" );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_RGBA32F_ARB, 16, 256,0,
                  GL_RGBA, GL_FLOAT,
                  edge_decode_normal.data() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    glGenTextures( 1, &m_tex_normal_flip );
    glBindTexture( GL_TEXTURE_2D, m_tex_normal_flip );
    log.setObjectLabel( GL_TEXTURE, m_tex_normal_flip, "flipped edge intersect normal table" );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_RGBA32F_ARB, 16, 256,0,
                  GL_RGBA, GL_FLOAT,
                  edge_decode_normal_flip.data() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    glBindTexture( GL_TEXTURE_2D, 0 );
}

HPMCIntersectingEdgeTable::~HPMCIntersectingEdgeTable()
{
    if( m_tex != 0 ) {
        glDeleteTextures( 1, &m_tex );
    }
    if( m_tex_flip != 0 ) {
        glDeleteTextures( 1, &m_tex_flip );
    }
    if( m_tex_normal != 0 ) {
        glDeleteTextures( 1, &m_tex_normal );
    }
    if( m_tex_normal_flip != 0 ) {
        glDeleteTextures( 1, &m_tex_normal_flip );
    }
}

} // of namespace glhpmc
