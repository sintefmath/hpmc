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

namespace cuhpmc {

// constant mem size: 64kb, cache working set: 8kb.
// Count + pad :  1+3 elements :    16 bytes :    16 bytes
// Level 0     :    4 elements :    16 bytes :    32 bytes
// Level 1     :   20 elements :    80 bytes :   112 bytes
// Level 2     :  100 elements :   400 bytes :   512 bytes
// Level 3     :  500 elements :  2000 bytes :  2112 bytes
// Level 4     : 2500 elements : 10000 bytes : 12112 bytes
// Levels 0-2: 32*4*4=512 bytes :
// Level  3:

__constant__ uint  hp5_const_offsets[32];



static
__device__
__forceinline__
void
downTraversalStep( uint& pos, uint& key, const uint4& val )
{
    pos *= 5;
    if( val.x <= key ) {
        pos++;
        key -=val.x;
        if( val.y <= key ) {
            pos++;
            key-=val.y;
            if( val.z <= key ) {
                pos++;
                key-=val.z;
                if( val.w <= key ) {
                    pos++;
                    key-=val.w;
                }
            }
        }
    }
}

static
__device__
__forceinline__
uint
__bitfieldextract( uint src, uint offset, uint bits )
{
    uint ret;
    asm volatile( "bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(src), "r"(offset), "r"(bits) );
    return ret;
}


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
        uint pos = 0;
        uint isec_off;
        {
            uint key = triangle;
            for(int l=0; l<max_level-3; l++) {
                // stored as 4 x 32 = 128 bits
                uint4 val = hp5_d[ hp5_const_offsets[l] + pos ];
                downTraversalStep( pos, key, val );
            }
            for(int l=max_level-3; l<max_level-1; l++) {
                // stored as 4 x 8 = 32 bits
                uint val = ((uint*)(hp5_d + hp5_const_offsets[ l ]))[pos];
                pos = 5*pos;
                uint t = __bitfieldextract( val, 0, 8 );
                if( t <= key ) {
                    pos++;
                    key -= t;
                    uint t = __bitfieldextract( val, 8, 8 );
                    if( t <= key ) {
                        pos++;
                        key -= t;
                        uint t = __bitfieldextract( val, 16, 8 );
                        if( t <= key ) {
                            pos++;
                            key -= t;
                            uint t = __bitfieldextract( val, 24, 8 );
                            if( t <= key ) {
                                pos++;
                                key -= t;
                            }
                        }
                    }
                }
            }
            {
                // stored as 4 x 4 = 16 bits
                uint val = ((unsigned short int*)(hp5_d + hp5_const_offsets[ max_level-1 ] ))[ pos ];
                pos = 5*pos;
                uint t = __bitfieldextract( val, 0, 4 );
                if( t <= key ) {
                    pos++;
                    key -= t;
                    t = __bitfieldextract( val, 4, 4 );
                    if( t <= key ) {
                        pos++;
                        key -= t;
                        t = __bitfieldextract( val, 8, 4 );
                        if( t <= key ) {
                            pos++;
                            key -= t;
                            t = __bitfieldextract( val, 12, 4 );
                            if( t <= key ) {
                                pos++;
                                key -= t;
                            }
                        }
                    }
                }
            }
            uint mc_case = mc_cases_d[pos];
            isec_off = 16*mc_case + 3*key;
        }

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

            assert( this_vertex_mask & vertices );

            const uint lower_vertices = vertices & (this_vertex_mask-1);

            uint index = __popc( lower_vertices );
            assert( index < 3 );

            // -- Up-traversal step 0
            {   // stored as 4 x 2 = 8 bits
                uint component = adjusted_pos % 5;
                adjusted_pos = adjusted_pos/5;
                unsigned char val_ = ((unsigned char*)(vertex_pyramid_d + hp5_const_offsets[ max_level-1 ] ))[ adjusted_pos ];

                if( component > 3 ) { index += __bitfieldextract(val_, 6, 2); }
                if( component > 2 ) { index += __bitfieldextract(val_, 4, 2); }
                if( component > 1 ) { index += __bitfieldextract(val_, 2, 2); }
                if( component > 0 ) { index += __bitfieldextract(val_, 0, 2); }
            }
            // -- Up-traversal step 1
            {   // stored as 4 x 4 = 16 bits
                uint component = adjusted_pos % 5;
                adjusted_pos = adjusted_pos/5;
                uint val_ = ((unsigned short int*)(vertex_pyramid_d + hp5_const_offsets[ max_level-2 ] ))[ adjusted_pos ];
                if( component > 3 ) { index += __bitfieldextract(val_,12, 4); }
                if( component > 2 ) { index += __bitfieldextract(val_, 8, 4); }
                if( component > 1 ) { index += __bitfieldextract(val_, 4, 4); }
                if( component > 0 ) { index += __bitfieldextract(val_, 0, 4); }
            }
            // -- Up-traversal step 2
            {   // stored as 4 x 8 = 32 bits
                uint component = adjusted_pos % 5;
                adjusted_pos = adjusted_pos/5;
                uchar4 val_ = ((uchar4*)(vertex_pyramid_d + hp5_const_offsets[ max_level-3 ]))[adjusted_pos];
                uint4 val = make_uint4( val_.x,
                                        val_.y,
                                        val_.z,
                                        val_.w );
                if( component > 3 ) { index += val.w; }
                if( component > 2 ) { index += val.z; }
                if( component > 1 ) { index += val.y; }
                if( component > 0 ) { index += val.x; }
            }
            for(int l=max_level-4; l>=0; l--) {
                uint component = adjusted_pos % 5;
                adjusted_pos = adjusted_pos/5;
                uint4 val = vertex_pyramid_d[ hp5_const_offsets[l] + adjusted_pos ];
                if( component > 3 ) { index += val.w; }
                if( component > 2 ) { index += val.z; }
                if( component > 1 ) { index += val.y; }
                if( component > 0 ) { index += val.x; }
            }
            indices[ 3*threadIdx.x + i ] = index;
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
    cudaMemcpyToSymbolAsync( hp5_const_offsets,
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
