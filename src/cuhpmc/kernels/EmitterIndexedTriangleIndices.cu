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
#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cassert>
#include <builtin_types.h>
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/FieldGlobalMemUChar.hpp>
#include <cuhpmc/IsoSurfaceIndexed.hpp>
#include <cuhpmc/EmitterTriIdx.hpp>
#include <cuhpmc/CUDAErrorException.hpp>
#include "HP5Traversal.hpp"

namespace cuhpmc {


__global__
void
TriangleIndicesKernel( uint* __restrict__                output_d,
                       const uint4* __restrict__         hp5_d,
                       const uint4* __restrict__         vertex_pyramid_d,
                       const unsigned char* __restrict__ mc_cases_d,
                       const unsigned char* __restrict__ case_indexed_intersect_edge_d,
                       const uint                        triangles,
                       const uint                        max_level,
                       const uint                        shift_y_chunk,  // = 800*chunks.x-4*5*32,
                       const uint                        shift_z_chunk ) // = 800*chunks.x*chunks.y-4
{
    __shared__ uint indices[ 256*3 ];

    uint triangle = 256*blockIdx.x + threadIdx.x;

    if( triangle < triangles ) {
        uint pos;
        uint key = triangle;
        trianglePyramidDownTraverse( pos, key, max_level, hp5_d );
        uint mc_case = mc_cases_d[pos];
        uint isec_off = 16*mc_case + 3*key;

        uint shift_mask = 0;
        {   // Determine if shifts cross chunk boundaries
            uint in_chunk_ix = pos % 800;
            if( ((in_chunk_ix / 5)%32) == 30 ) { shift_mask = 1; }
            if( (in_chunk_ix/5)/32 == 4 ) { shift_mask += 2;  }
            if( in_chunk_ix%5 == 4 ) {  shift_mask += 4; }
        }

        for(uint i=0; i<3; i++) {
            uint edge_flags = case_indexed_intersect_edge_d[ isec_off + i ];
            uint adjusted_pos = pos;

            if( edge_flags&(1<<5) ) {
                adjusted_pos += ( shift_mask&0x1 ? 800-30*5 : 5 );
            }
            if( edge_flags&(1<<6) ) {
                adjusted_pos += ( shift_mask&0x2 ? shift_y_chunk : 5*32 );
            }
            if( edge_flags&(1<<7) ) {
                adjusted_pos += ( shift_mask&0x4 ? shift_z_chunk : 1 );
            }

            uint cell_case = mc_cases_d[ adjusted_pos ];

            // Determine # edge intersections in this cell
            const uint vertex_0___mask = (cell_case&0x1 ? 0x16 : 0);    // %xxxxxxx1 ? %00010110 : %00000000
            const uint vertex_124_mask = cell_case & 0x16;              // %000x0xx0 = %xxxxxxxx
            const uint vertices = vertex_0___mask ^ vertex_124_mask;    //

            // Extract edge bit for this one
            const uint this_vertex_mask = edge_flags & 0x16;
            // Mask out edge intersections before the current intersection

            const uint lower_vertices = vertices & (this_vertex_mask-1);

            indices[ 3*threadIdx.x + i ] = __popc( lower_vertices )
                                         + vertexPyramidUpTraverse( adjusted_pos,
                                                                    max_level,
                                                                    vertex_pyramid_d );;
        }
    }
    __syncthreads();

    output_d[ 3*256*blockIdx.x +   0 + threadIdx.x ] = indices[ threadIdx.x + 0   ];
    output_d[ 3*256*blockIdx.x + 256 + threadIdx.x ] = indices[ threadIdx.x + 256 ];
    output_d[ 3*256*blockIdx.x + 512 + threadIdx.x ] = indices[ threadIdx.x + 512 ];
}

void
EmitterTriIdx::invokeTriangleIndicesKernel( unsigned int* output_d, uint tris, cudaStream_t stream )
{
    if( tris == 0 ) {
        return;
    }
    //tris = tris;
    cudaMemcpyToSymbolAsync( triangle_hp5_offsets,
                             m_iso_surface->hp5LevelOffsetsDev(),
                             sizeof(uint)*32,
                             0,
                             cudaMemcpyDeviceToDevice,
                             stream );
    cudaMemcpyToSymbolAsync( vertex_hp5_offsets,
                             m_iso_surface->hp5LevelOffsetsDev(),
                             sizeof(uint)*32,
                             0,
                             cudaMemcpyDeviceToDevice,
                             stream );

    dim3 gs( ((tris+255)/256), 1, 1 );
    dim3 bs( 256, 1, 1 );

    TriangleIndicesKernel<<<gs,bs,0,stream>>>( output_d,
                                               m_iso_surface->trianglePyramidDev(),
                                               m_iso_surface->vertexPyramidDev(),
                                               m_iso_surface->mcCasesDev(),
                                               m_constants->caseIndexIntersectEdgeDev(),
                                               tris,
                                               m_iso_surface->hp5Levels(),
                                               800*m_iso_surface->hp5Chunks().x-4*5*32,
                                               800*m_iso_surface->hp5Chunks().x*m_iso_surface->hp5Chunks().y-4 );

    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }
}




} // of namespace cuhpmc
