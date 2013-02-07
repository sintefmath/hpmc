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
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/Tables.hpp>

namespace cuhpmc {


Constants::Constants()
    : m_triangle_count_dev( NULL ),
      m_case_intersect_edge_tex(0)
{
    unsigned char vtxtricnt[256];
    unsigned char tricnt[256];
    unsigned char eisec[256*16];
    unsigned char ixisec[256*16];

    for(uint j=0; j<256; j++) {
        // case bits:
        // bit 0: p000 is inside
        // bit 1: p100 is inside
        // bit 2: p010 is inside
        // bit 3: p110 is inside
        // bit 4: p001 is inside
        // bit 5: p101 is inside
        // bit 6: p011 is inside
        // bit 7: p111 is inside

        const uint mask = 0x16; // 00010110
        uint t0 = (j&0x1==1?mask:0);
        uint t1 = j & mask;
        uint t2 = t0 ^ t1;
        uint vtx_cnt = (t2>>1) - (t2>>2) - (t2>>3) - (t2>>4);  // popcnt_4


        /*

        std::cerr << "case " << j << "\tcasebits="
                  << (j&128?'+':'-')
                  << (j&64?'+':'-')
                  << (j&32?'+':'-')
                  << (j&16?'Z':'z')
                  << (j&8?'+':'-')
                  << (j&4?'Y':'y')
                  << (j&2?'X':'x')
                  << (j&1?'O':'o')
                  << ", axis isecs="
                  << (t2&128?'+':'-')
                  << (t2&64?'+':'-')
                  << (t2&32?'+':'-')
                  << (t2&16?'Z':'-')
                  << (t2&8?'+':'-')
                  << (t2&4?'Y':'-')
                  << (t2&2?'X':'-')
                  << (t2&1?'+':'-')
                  << ", # isecs="
                  << s << "\n";

        // Find which bit is the nth bit
        for(int k=0; k<s; k++ ) {
            const uint mask = 0x16; // 00010110
            uint t0 = (j&0x1==1?mask:0);
            uint t1 = j & mask;
            uint t2 = t0 ^ t1;

            uint t3 = t2;
            t3 = (k>0? (t3&(t3-1)) : t3 );  // conditionally kill rightmost bit
            t3 = (k>1? (t3&(t3-1)) : t3 );  // conditionally kill rightmost bit
            t3 = t3 & (-t3);                // isolate rightmost bit


            // Then, given a bit in t3 and edge mask in t2, which index is it?
            uint t4 = t2 & (t3-1); // mask out all bits before t3
            uint l = (t4>>1) - (t4>>2);  // popcnt_2(x>>1)
            std::cerr << "  axis="
                      << (t3&128?'+':'-')
                      << (t3&64?'+':'-')
                      << (t3&32?'+':'-')
                      << (t3&16?'Z':'-')
                      << (t3&8?'+':'-')
                      << (t3&4?'Y':'-')
                      << (t3&2?'X':'-')
                      << (t3&1?'+':'-')
                      << ", done="
                      << (t4&128?'+':'-')
                      << (t4&64?'+':'-')
                      << (t4&32?'+':'-')
                      << (t4&16?'z':'-')
                      << (t4&8?'+':'-')
                      << (t4&4?'y':'-')
                      << (t4&2?'x':'-')
                      << (t4&1?'+':'-')
                      << ", off="
                      << l
                      << "\n";


        }

*/


        for(uint i=0; i<16; i++) {
            int m = triangle_table[ j ][ i ];
            if( triangle_table[j][i] == -1 ) {
                vtxtricnt[j] =  (vtx_cnt<<5) | (i/3);
                tricnt[j] = (i/3);
                break;
            }
            else {
                int a = edge_table[m][0];
                int b = edge_table[m][1];
                eisec[16*j+i] = a | (b<<3);
                ixisec[16*j+i] = edge_table[m][2];
            }
        }
    }

    // copy tri cnt table to device
    if( cudaMalloc( (void**)(&m_vertex_triangle_count_dev), sizeof(vtxtricnt) ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }
    if( cudaMemcpy( m_vertex_triangle_count_dev, vtxtricnt, sizeof(vtxtricnt), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        cudaFree( m_vertex_triangle_count_dev );
        m_vertex_triangle_count_dev = NULL;
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }

    // copy tri cnt table to device
    if( cudaMalloc( (void**)(&m_triangle_count_dev), sizeof(tricnt) ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }
    if( cudaMemcpy( m_triangle_count_dev, tricnt, sizeof(tricnt), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        cudaFree( m_triangle_count_dev );
        m_triangle_count_dev = NULL;
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }

    // copy edge intersection table to device
    if( cudaMalloc( (void**)(&m_case_intersect_edge_d), sizeof(eisec) ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }
    if( cudaMemcpy( m_case_intersect_edge_d, eisec, sizeof(eisec), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }

    if( cudaMalloc( (void**)(&m_case_indexed_intersect_edge_d), sizeof(ixisec) ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }
    if( cudaMemcpy( m_case_indexed_intersect_edge_d, ixisec, sizeof(ixisec), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }

//    unsigned char*  m_case_indexed_intersect_edge_d;



#ifdef ENABLE_CUHPMC_INTEROP
    glGenTextures( 1, &m_case_intersect_edge_tex );
    glBindTexture( GL_TEXTURE_2D, m_case_intersect_edge_tex );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_R8UI, 16, 256, 0,
                  GL_RED_INTEGER, GL_UNSIGNED_BYTE,
                  eisec );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_2D, 0 );
#endif

}

Constants::~Constants()
{
    if( m_triangle_count_dev != NULL ) {
        cudaFree( m_triangle_count_dev );
    }
#ifdef ENABLE_CUHPMC_INTEROP
    glDeleteTextures( 1, &m_case_intersect_edge_tex );
#endif
}

} // of namespace cuhpmc
