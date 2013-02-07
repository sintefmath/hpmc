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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <builtin_types.h>
#include <vector_functions.h>
#include <iostream>
#include <stdexcept>
#include <cuhpmc/IsoSurfaceGL.hpp>
#include <cuhpmc/FieldGLBufferUChar.hpp>
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/CUDAErrorException.hpp>

namespace cuhpmc {

IsoSurfaceGL::IsoSurfaceGL( FieldGL *field )
    : IsoSurface( field ),
      m_field_gl( field )
{
    glGenBuffers( 1, &m_hp5_hp_buf );
    glBindBuffer( GL_TEXTURE_BUFFER, m_hp5_hp_buf );
    glBufferData( GL_TEXTURE_BUFFER,
                  4*sizeof(uint)* m_hp5_size,
                  NULL,
                  GL_DYNAMIC_COPY );
    glGenTextures( 1, &m_hp5_gl_tex );
    glBindTexture( GL_TEXTURE_BUFFER, m_hp5_gl_tex );
    glTexBuffer( GL_TEXTURE_BUFFER, GL_RGBA32UI, m_hp5_hp_buf );
    glBindTexture( GL_TEXTURE_BUFFER, 0 );

    glGenBuffers( 1, &m_case_buf );
    glBindBuffer( GL_TEXTURE_BUFFER, m_case_buf );
    glBufferData( GL_TEXTURE_BUFFER,
                  sizeof(unsigned char)* m_hp5_input_N,
                  NULL,
                  GL_DYNAMIC_COPY );
    glBindBuffer( GL_TEXTURE_BUFFER, 0 );

    glGenTextures( 1, &m_case_gl_tex );
    glBindTexture( GL_TEXTURE_BUFFER, m_case_gl_tex );
    glTexBuffer( GL_TEXTURE_BUFFER, GL_R8UI, m_case_buf );
    glBindTexture( GL_TEXTURE_BUFFER, 0 );

    cudaGraphicsGLRegisterBuffer( &m_resources[0], m_hp5_hp_buf, cudaGraphicsRegisterFlagsWriteDiscard );
    cudaGraphicsGLRegisterBuffer( &m_resources[1], m_case_buf, cudaGraphicsRegisterFlagsWriteDiscard  );

    /*
    cudaMalloc( (void**)&m_hp5_hp_d, 4*sizeof(uint)* m_hp5_size );
    cudaMalloc( (void**)&m_case_d, m_hp5_input_N );

    cudaMalloc( (void**)&m_hp5_offsets_d, sizeof(uint)*32 );
    cudaMemset( m_hp5_offsets_d, 0, sizeof(uint)*32 );
    cudaMemcpy( m_hp5_offsets_d, m_hp5_offsets.data(), sizeof(uint)*m_hp5_offsets.size(), cudaMemcpyHostToDevice );
*/
}

IsoSurfaceGL::~IsoSurfaceGL( )
{
    cudaGraphicsUnregisterResource( m_resources[0] );
    cudaGraphicsUnregisterResource( m_resources[1] );
    glDeleteBuffers( 1, &m_hp5_hp_buf );
    glDeleteBuffers( 1, &m_case_buf );
}


void
IsoSurfaceGL::build( float iso, cudaStream_t stream )
{
    cudaError_t error;
    cudaGraphicsResource* resources[3] = {
        m_resources[0],
        m_resources[1],
        m_field_gl->resource()
    };


    m_iso = iso;
    m_hp5_top_h[0] = 0;
    uint3 field_size = make_uint3( m_field->width(), m_field->height(), m_field->depth() );

    error = cudaGraphicsMapResources( 3, resources, stream );
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }
    try {
        uint4* hp5_hp_d = NULL;
        size_t hp5_hp_size = 0;
        error = cudaGraphicsResourceGetMappedPointer( (void**)&hp5_hp_d, &hp5_hp_size, resources[0] );
        if( error != cudaSuccess ) {
            throw CUDAErrorException( error );
        }

        unsigned char* case_d = NULL;
        size_t case_size = 0;
        error = cudaGraphicsResourceGetMappedPointer( (void**)&case_d, &case_size, resources[1] );
        if( error != cudaSuccess ) {
            throw CUDAErrorException( error );
        }

        if( FieldGLBufferUChar* field = dynamic_cast<FieldGLBufferUChar*>( m_field_gl ) ) {
            unsigned char* field_ptr = NULL;
            size_t field_ptr_size = 0;

            error = cudaGraphicsResourceGetMappedPointer( (void**)&field_ptr, &field_ptr_size, resources[2] );
            if( error != cudaSuccess ) {
                throw CUDAErrorException( error );
            }

            invokeBaseBuildup( hp5_hp_d + m_hp5_offsets[ m_hp5_levels-3 ],
                               m_hp5_sb_d + m_hp5_offsets[ m_hp5_levels-3 ],
                               m_hp5_level_sizes[ m_hp5_levels-1 ],
                               hp5_hp_d + m_hp5_offsets[ m_hp5_levels-2 ],
                               hp5_hp_d + m_hp5_offsets[ m_hp5_levels-1 ],
                               case_d,
                               m_iso,
                               m_hp5_chunks,
                               field_ptr,
                               field_size,
                               m_constants->triangleCountDev(),
                               stream );

        }
        else {
            throw std::runtime_error( "Unsupported field type" );
        }

        for( uint i=m_hp5_first_triple_level; i>m_hp5_first_double_level; i-=2 ) {
            invokeDoubleBuildup(  hp5_hp_d + m_hp5_offsets[i-2],
                                  m_hp5_sb_d + m_hp5_offsets[i-2],
                                  hp5_hp_d + m_hp5_offsets[i-1],
                                  m_hp5_sb_d + m_hp5_offsets[i],
                                  m_hp5_level_sizes[i-1],
                                  stream );
        }
        for( uint i=m_hp5_first_double_level; i>m_hp5_first_single_level; --i ) {
            invokeSingleBuildup( hp5_hp_d   + m_hp5_offsets[ i-1 ],
                                          m_hp5_sb_d + m_hp5_offsets[ i-1 ],
                                          m_hp5_sb_d + m_hp5_offsets[ i   ],
                                          m_hp5_level_sizes[i-1],
                                          stream );
        }
        invokeApexBuildup( m_hp5_top_d,
                           hp5_hp_d,
                           m_hp5_sb_d + 32,
                           m_hp5_level_sizes[2],
                           stream );

    }
    catch( ... ) {
        error = cudaGraphicsUnmapResources( 3, resources, stream );
        if( error != cudaSuccess ) {
            throw CUDAErrorException( error );
        }
        throw;
    }
    error = cudaGraphicsUnmapResources( 3, resources, stream );
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }


}




} // of namespace cuhpmc
