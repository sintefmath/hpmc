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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/Tables.hpp>

namespace cuhpmc {


Constants::Constants()
    : m_vtxcnt_dev( NULL )
{
    unsigned char vtxcnt[256];
    unsigned char eisec[256*16];

    for(uint j=0; j<256; j++) {
        for(uint i=0; i<16; i++) {
            int m = triangle_table[ j ][ i ];
            if( triangle_table[j][i] == -1 ) {
                vtxcnt[j] = i;
                break;
            }
            else {
                int a = edge_table[m][0];
                int b = edge_table[m][1];
                eisec[16*j+i] = a | (b<<3);
            }
        }
    }

    if( cudaMalloc( (void**)(&m_vtxcnt_dev), sizeof(vtxcnt) ) != cudaSuccess ) {
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }
    if( cudaMemcpy( m_vtxcnt_dev, vtxcnt, sizeof(vtxcnt), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        cudaFree( m_vtxcnt_dev );
        m_vtxcnt_dev = NULL;
        throw std::runtime_error( std::string( cudaGetErrorString( cudaGetLastError() ) ) );
    }
}

Constants::~Constants()
{
    if( m_vtxcnt_dev != NULL ) {
        cudaFree( m_vtxcnt_dev );
    }
}

} // of namespace cuhpmc
