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
#include <assert.h>
#include <cuda.h>
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
hp5PosToCellPos( int3&                              i0,
                 uint&                              mc_case,
                 const uint                         pos,
                 const uint3&                       chunks,
                 const unsigned char* __restrict__  mc_cases_d )
{
    uint c_lix = pos / 800u;
    uint t_lix = pos % 800u;
    int3 ci = make_int3( 31*( c_lix % chunks.x ),
                          5*( (c_lix/chunks.x) % chunks.y ),
                          5*( (c_lix/chunks.x) / chunks.y ) );

    // calc 3D pos within cunk
    i0 = make_int3( ci.x + ((t_lix / 5)%32),
                    ci.y + ((t_lix / 5)/32),
                    ci.z + ( t_lix%5 ) );

    mc_case = mc_cases_d[ pos ];
}


__global__
void
VertexN3FV3Fkernel( float* __restrict__               output_d,
                    const uint4* __restrict__         vertex_pyramid_d,
                    const unsigned char* __restrict__ mc_cases_d,
                    const unsigned char* __restrict__ case_intersect_edge_d,
                    const uint3                       chunks,
                    const int3                        cells,
                    const uint                        vertices,
                    const uint                        max_level,
                    const float                       iso,
                    const unsigned char*              field_d,
                    const uint                        field_row_pitch,
                    const uint                        field_slice_pitch,
                    const float3                      scale )
{

    uint vertex = 256*blockIdx.x + threadIdx.x;
    if( vertex < vertices ) {

        uint key = vertex;
        uint pos = 0;
        int l = 0;
        for(; l<max_level-3; l++) {
            uint4 val = vertex_pyramid_d[ hp5_const_offsets[l] + pos ];
            downTraversalStep( pos, key, val );
        }
        {   // second reduction is 4 x 8 bits = 32 bits
            uchar4 val_ = ((uchar4*)(vertex_pyramid_d + hp5_const_offsets[ max_level-3 ]))[pos];
            uint4 val = make_uint4( val_.x,
                                    val_.y,
                                    val_.z,
                                    val_.w );
            downTraversalStep( pos, key, val );
        }
        {   // first reduction is 4 x 4 bits = 16 bits
            short1 val_ = ((short1*)(vertex_pyramid_d + hp5_const_offsets[ max_level-2 ] ))[ pos ];
            uint4 val = make_uint4( (val_.x   )  & 0xfu,
                                    (val_.x>>4)  & 0xfu,
                                    (val_.x>>8)  & 0xfu,
                                    (val_.x>>12) & 0xfu );
            downTraversalStep( pos, key, val );
        }
        {   // base layer is 4 x 2 bits = 8 bits
            unsigned char val_ = ((unsigned char*)(vertex_pyramid_d + hp5_const_offsets[ max_level-1 ] ))[ pos ];
            uint4 val = make_uint4( (val_   ) & 0x3u,
                                    (val_>>2) & 0x3u,
                                    (val_>>4) & 0x3u,
                                    (val_>>6) & 0x3u );
            downTraversalStep( pos, key, val );
        }

        int3 i0;
        uint mc_case;
        hp5PosToCellPos( i0, mc_case, pos, chunks, mc_cases_d );

        // extract edge intersection case from vertex case
        const uint mask = 0x16u;    // =%00010110
        const uint t0 = (mc_case&0x1==1?mask:0u);
        uint edge = (mc_case ^ t0 ) & mask;

        // strip 1, or 2 bits from the right
        if( key > 0 ) {
            edge = edge & (edge-1);
        }
        if( key > 1 ) {
            edge = edge & (edge-1);
        }
        // isolate rightmost bit
        edge = edge & (-edge);

        // Determine other end of edge
        int3 i1 = make_int3( i0.x + ((edge>>1) & 0x1u),
                             i0.y + ((edge>>2) & 0x1u),
                             i0.z + ((edge>>4) & 0x1u) );

        /*        assert( i0.x < 128 );
        assert( i0.y < 128 );
        assert( i0.z < 128 );

        assert( i1.x < 128 );
        assert( i1.y < 128 );
        assert( i1.z < 128 );
*/
        uint i0_ix = i0.x
                   + i0.y*field_row_pitch
                   + i0.z*field_slice_pitch;

        uint i1_ix = i1.x
                   + i1.y*field_row_pitch
                   + i1.z*field_slice_pitch;


        float f0 = field_d[ i0_ix ];
        float f0_x = field_d[ i0_ix + (i0.x < cells.x ? 1 : 0) ]-f0;
        float f0_y = field_d[ i0_ix + (i0.y < cells.y ? field_row_pitch : 0) ]-f0;
        float f0_z = field_d[ i0_ix + (i0.z < cells.z ? field_slice_pitch : 0) ]-f0;


        float f1 = field_d[ i1_ix ];
        float f1_x = field_d[ i1_ix + (i1.x < cells.x ? 1 : 0 )]-f1;
        float f1_y = field_d[ i1_ix + (i1.y < cells.y ? field_row_pitch : 0 ) ]-f1;
        float f1_z = field_d[ i1_ix + (i1.z < cells.z ? field_slice_pitch : 0 ) ]-f1;

        float t = 0.5f;//(iso-f0)/(f1-f0);
        float s = 1.f-t;

        float n_x = s*f0_x + t*f1_x;
        float n_y = s*f0_y + t*f1_y;
        float n_z = s*f0_z + t*f1_z;

        output_d[ 6*vertex + 0 ] = n_x;
        output_d[ 6*vertex + 1 ] = n_y;
        output_d[ 6*vertex + 2 ] = n_z;
        output_d[ 6*vertex + 3 ] = scale.x*(s*i0.x + t*i1.x);
        output_d[ 6*vertex + 4 ] = scale.y*(s*i0.y + t*i1.y);
        output_d[ 6*vertex + 5 ] = scale.z*(s*i0.z + t*i1.z);
    }
}

void
EmitterTriIdx::invokeVertexN3FV3Fkernel( float* output_d, uint vtx, cudaStream_t stream )
{
    if( vtx == 0 ) {
        return;
    }
    cudaMemcpyToSymbolAsync( hp5_const_offsets,
                             m_iso_surface->hp5LevelOffsetsDev(),
                             sizeof(uint)*32,
                             0,
                             cudaMemcpyDeviceToDevice,
                             stream );

    dim3 gs( ((vtx+255)/256), 1, 1 );
    dim3 bs( 256, 1, 1 );
    if( FieldGlobalMemUChar* field = dynamic_cast<FieldGlobalMemUChar*>( m_field ) ) {
        VertexN3FV3Fkernel<<<gs,bs,0,stream>>>( output_d,
                                                m_iso_surface->vertexPyramidDev(),
                                                m_iso_surface->mcCasesDev(),
                                                m_constants->caseIntersectEdgeDev(),
                                                m_iso_surface->hp5Chunks(),
                                                make_int3( m_iso_surface->cells().x,
                                                           m_iso_surface->cells().y,
                                                           m_iso_surface->cells().z ),
                                                vtx,
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

#if 0
void
EmitterTriIdx::invokeKernel( float* output_d, uint tris, cudaStream_t stream )
{
    if( tris == 0 ) {
        return;
    }
    tris = tris;
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
                                           make_int3( m_is)
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
#endif



} // of namespace cuhpmc
