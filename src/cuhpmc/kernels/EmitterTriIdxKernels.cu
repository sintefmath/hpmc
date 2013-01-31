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
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/FieldGlobalMemUChar.hpp>
#include <cuhpmc/IsoSurfaceIndexed.hpp>
#include <cuhpmc/EmitterTriIdx.hpp>

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
zindex_writer( float* __restrict__               output_d,
              const uint4* __restrict__         hp5_d,
              const unsigned char* __restrict__ mc_cases_d,
              const unsigned char* __restrict__ case_intersect_edge_d,
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
        for(; l<max_level-2; l++) {
            uint4 val = hp5_d[ hp5_const_offsets[l] + pos ];
            downTraversalStep( pos, key, val );
        }
        {   // l1 -> fetch lower byte of 16-bit value
            uint4 val = hp5_d[ hp5_const_offsets[ max_level-2 ] + pos ];
            val.x = val.x & 0xffu;
            val.y = val.y & 0xffu;
            val.z = val.z & 0xffu;
            val.w = val.w & 0xffu;
            downTraversalStep( pos, key, val );
        }
        {   // l0 -> fetch lower nibble of 8-bit value
            uint4 val = hp5_d[ hp5_const_offsets[ max_level-1 ] + pos ];
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
EmitterTriIdx::invokeKernel( float* output_d, uint tris, cudaStream_t stream )
{
    if( tris == 0 ) {
        return;
    }
    cudaMemcpyToSymbolAsync( hp5_const_offsets,
                             m_iso_surface->hp5LevelOffsetsDev(),
                             sizeof(uint)*32,
                             0,
                             cudaMemcpyDeviceToDevice,
                             stream );

    dim3 gs( ((tris+255)/256), 1, 1 );
    dim3 bs( 256, 1, 1 );

    if( FieldGlobalMemUChar* field = dynamic_cast<FieldGlobalMemUChar*>( m_field ) ) {
        zindex_writer<<<gs,bs,0,stream>>>( output_d,
                                           m_iso_surface->trianglePyramidDev(),
                                           m_iso_surface->mcCasesDev(),
                                           m_constants->caseIntersectEdgeDev(),
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
}




} // of namespace cuhpmc
