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
#include <builtin_types.h>
#include <cassert>
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

__device__
__forceinline__
void
hp5PosToCellPos( uint3&                             i0,
                 uint&                              mc_case,
                 const uint                         pos,
                 const uint3&                       chunks,
                 const unsigned char* __restrict__  mc_cases_d )
{
    uint c_lix = pos / 800u;
    uint t_lix = pos % 800u;
    uint3 ci = make_uint3( 31*( c_lix % chunks.x ),
                           5*( (c_lix/chunks.x) % chunks.y ),
                           5*( (c_lix/chunks.x) / chunks.y ) );

    // calc 3D pos within cunk
    i0 = make_uint3( ci.x + ((t_lix / 5)%32),
                     ci.y + ((t_lix / 5)/32),
                     ci.z + ( t_lix%5 ) );

    mc_case = mc_cases_d[ pos ];
}


__global__
void
TriangleIndicesKernel( float* __restrict__               output_d,
                       const uint4* __restrict__         hp5_d,
                       const uint4* __restrict__         vertex_pyramid_d,
                       const unsigned char* __restrict__ mc_cases_d,
                       const unsigned char* __restrict__ case_intersect_edge_d,
                       const unsigned char* __restrict__ case_indexed_intersect_edge_d,
                       const uint3                       chunks,
                       const uint                        triangles,
                       const uint                        max_level,
                       const float                       iso,
                       const unsigned char*              field_d,
                       const uint                        field_row_pitch,
                       const uint                        field_slice_pitch,
                       const float3                      scale )
{

    uint triangle = 256*blockIdx.x + threadIdx.x;

    if( triangle < triangles ) {

        uint key = triangle;
        uint pos = 0;
        int l = 0;
        for(; l<max_level-3; l++) {
            uint4 val = hp5_d[ hp5_const_offsets[l] + pos ];
            downTraversalStep( pos, key, val );
        }
        for(; l<max_level-1; l++) {
            // l1 -> fetch lower byte of 16-bit value
            uchar4 val_ = ((uchar4*)(hp5_d + hp5_const_offsets[ l ]))[pos];
            uint4 val = make_uint4( val_.x,
                                    val_.y,
                                    val_.z,
                                    val_.w );
            /*
            hp5_d[ hp5_const_offsets[ max_level-2 ] + pos ];
            val.x = val.x & 0xffu;
            val.y = val.y & 0xffu;
            val.z = val.z & 0xffu;
            val.w = val.w & 0xffu;*/
            downTraversalStep( pos, key, val );
        }
        {   // l0 -> fetch lower nibble of 8-bit value
            short1 val_ = ((short1*)(hp5_d + hp5_const_offsets[ max_level-1 ] ))[ pos ];
            uint4 val = make_uint4( val_.x & 0xfu,
                                    (val_.x>>4) & 0xfu,
                                    (val_.x>>8) & 0xfu,
                                    (val_.x>>12) & 0xfu );

           // uint4 val = hp5_d[ hp5_const_offsets[ max_level-1 ] + pos ];
            val.x = val.x & 0xfu;
            val.y = val.y & 0xfu;
            val.z = val.z & 0xfu;
            val.w = val.w & 0xfu;
            downTraversalStep( pos, key, val );
        }
        uint rem = 3*key;

        uint3 i0;
        uint mc_case;
        hp5PosToCellPos( i0, mc_case, pos, chunks, mc_cases_d );

        for(uint i=0; i<3; i++) {
            uint edge_flags = case_indexed_intersect_edge_d[ 16*mc_case + rem + i ];
            // adjust for shift
            int3 cell = make_int3( i0.x + (((edge_flags&(1<<5)) ? 1 : 0  )),
                                   i0.y + (((edge_flags&(1<<6)) ? 1 : 0  )),
                                   i0.z + (((edge_flags&(1<<7)) ? 1 : 0  )) );
            int3 chunk = make_int3( cell.x / 31,
                                    cell.y / 5,
                                    cell.z / 5 );
            int3 sub_chunk = make_int3( cell.x % 31,
                                        cell.y % 5,
                                        cell.z % 5 );
            int ix = 800*( chunk.x + chunks.x*( chunk.y + chunks.y * chunk.z))
                   + sub_chunk.z + 5*(sub_chunk.x + 32*sub_chunk.y);

            uint cell_case = mc_cases_d[ ix ];

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

            // -- Up-traversal step 0
            {
                uint component = pos % 5;
                pos = pos/5;
                unsigned char val_ = ((unsigned char*)(vertex_pyramid_d + hp5_const_offsets[ max_level-1 ] ))[ pos ];
                uint4 val = make_uint4( (val_   ) & 0x3u,
                                        (val_>>2) & 0x3u,
                                        (val_>>4) & 0x3u,
                                        (val_>>6) & 0x3u );
                if( component > 3 ) {
                    index += val.w;
                }
                if( component > 2 ) {
                    index += val.z;
                }
                if( component > 1 ) {
                    index += val.y;
                }
                if( component > 0 ) {
                    index += val.x;
                }
            }
            // -- Up-traversal step 1
            {
                uint component = pos % 5;
                pos = pos/5;
                short1 val_ = ((short1*)(vertex_pyramid_d + hp5_const_offsets[ max_level-2 ] ))[ pos ];
                uint4 val = make_uint4( (val_.x   )  & 0xfu,
                                        (val_.x>>4)  & 0xfu,
                                        (val_.x>>8)  & 0xfu,
                                        (val_.x>>12) & 0xfu );
                if( component > 3 ) {
                    index += val.w;
                }
                if( component > 2 ) {
                    index += val.z;
                }
                if( component > 1 ) {
                    index += val.y;
                }
                if( component > 0 ) {
                    index += val.x;
                }
            }
            // -- Up-traversal step 2
            {
                uint component = pos % 5;
                pos = pos/5;
                uchar4 val_ = ((uchar4*)(vertex_pyramid_d + hp5_const_offsets[ l ]))[pos];
                uint4 val = make_uint4( val_.x,
                                        val_.y,
                                        val_.z,
                                        val_.w );
                if( component > 3 ) {
                    index += val.w;
                }
                if( component > 2 ) {
                    index += val.z;
                }
                if( component > 1 ) {
                    index += val.y;
                }
                if( component > 0 ) {
                    index += val.x;
                }
            }
            for(l=max_level-4; l>0; l--) {
                uint component = pos % 5;
                pos = pos/5;
                uint4 val = vertex_pyramid_d[ hp5_const_offsets[l] + pos ];
                if( component > 3 ) {
                    index += val.w;
                }
                if( component > 2 ) {
                    index += val.z;
                }
                if( component > 1 ) {
                    index += val.y;
                }
                if( component > 0 ) {
                    index += val.x;
                }
            }
        }


        /*
        uint c_lix = pos / 800u;
        uint t_lix = pos % 800u;
        uint3 ci = make_uint3( 31*( c_lix % chunks.x ),
                               5*( (c_lix/chunks.x) % chunks.y ),
                               5*( (c_lix/chunks.x) / chunks.y ) );

        // calc 3D pos within cunk
        i0 = make_uint3( ci.x + ((t_lix / 5)%32),
                         ci.y + ((t_lix / 5)/32),
                         ci.z + ( t_lix%5 ) );

        mc_case = mc_cases_d[ pos ];
        */
        for(uint i=0; i<3; i++ ) {
            uint isec = case_intersect_edge_d[ 16*mc_case + rem + i ];

            uint3 oa = make_uint3( i0.x + ((isec   )&1u),
                                   i0.y + ((isec>>1u)&1u),
                                   i0.z + ((isec>>2u)&1u) );
            uint oa_ix = oa.x
                       + oa.y*field_row_pitch
                       + oa.z*field_slice_pitch;
            float fa = field_d[ oa_ix ];
            float fa_x = field_d[ oa_ix + 1 ]-fa;
            float fa_y = field_d[ oa_ix + field_row_pitch ]-fa;
            float fa_z = field_d[ oa_ix + field_slice_pitch ]-fa;

            uint3 ob = make_uint3( i0.x + ((isec>>3u)&1u),
                                   i0.y + ((isec>>4u)&1u),
                                   i0.z + ((isec>>5u)&1u) );
            uint ob_ix = ob.x
                       + ob.y*field_row_pitch
                       + ob.z*field_slice_pitch;
            float fb = field_d[ ob_ix ];
            float fb_x = field_d[ ob_ix + 1 ]-fb;
            float fb_y = field_d[ ob_ix + field_row_pitch ]-fb;
            float fb_z = field_d[ ob_ix + field_slice_pitch ]-fb;

            float t = (iso-fa)/(fb-fa);
            float s = 1.f-t;

            float n_x = s*fa_x + t*fb_x;
            float n_y = s*fa_y + t*fb_y;
            float n_z = s*fa_z + t*fb_z;


            uint vtx = 3*triangle + i;

            output_d[ 6*vtx + 0 ] = n_x;
            output_d[ 6*vtx + 1 ] = n_y;
            output_d[ 6*vtx + 2 ] = n_z;
            output_d[ 6*vtx + 3 ] = scale.x*(s*oa.x + t*ob.x);
            output_d[ 6*vtx + 4 ] = scale.y*(s*oa.y + t*ob.y);
            output_d[ 6*vtx + 5 ] = scale.z*(s*oa.z + t*ob.z);
        }
    }
}

void
EmitterTriIdx::invokeTriangleIndicesKernel( float* output_d, uint tris, cudaStream_t stream )
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

    if( FieldGlobalMemUChar* field = dynamic_cast<FieldGlobalMemUChar*>( m_field ) ) {
        TriangleIndicesKernel<<<gs,bs,0,stream>>>( output_d,
                                                   m_iso_surface->trianglePyramidDev(),
                                                   m_iso_surface->vertexPyramidDev(),
                                                   m_iso_surface->mcCasesDev(),
                                                   m_constants->caseIntersectEdgeDev(),
                                                   m_constants->caseIndexIntersectEdgeDev(),
                                                   m_iso_surface->hp5Chunks(),
                                                   tris,
                                                   m_iso_surface->hp5Levels(),
                                                   256.f*m_iso_surface->iso(),
                                                   field->fieldDev(),
                                                   field->width(),
                                                   field->width()*field->height(),
                                                   make_float3( 1.f/(field->width()-1.f),
                                                                1.f/(field->height()-1.f),
                                                                1.f/(field->depth()-1.f) ) );
    }
    else {
        throw std::runtime_error( "EmitterTriIdx::invokeKernel: unsupported field type" );
    }
    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }
}




} // of namespace cuhpmc
