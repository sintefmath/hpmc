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
#include <cuda_gl_interop.h>
#include <cuhpmc/cuhpmc.hpp>
#include <cuhpmc/CUDAErrorException.hpp>
#include <cuhpmc/FieldGLBufferUChar.hpp>

namespace cuhpmc {

FieldGLBufferUChar::FieldGLBufferUChar( Constants*     constants,
                                        GLuint         field_buf,
                                        uint           width,
                                        uint           height,
                                        uint           depth )
    : Field( constants, width, height, depth ),
      m_mapped(false ),
      m_field_buf( field_buf ),
      m_field_resource( NULL )
{
    if( m_field_buf == 0 ) {
        throw std::runtime_error( "field_buf == 0" );
    }
    glGenTextures( 1, &m_field_gl_tex );
    glBindTexture( GL_TEXTURE_BUFFER, m_field_gl_tex );
    glTexBuffer( GL_TEXTURE_BUFFER, GL_R8, m_field_buf );
    glBindTexture( GL_TEXTURE_BUFFER, 0 );

    cudaGraphicsGLRegisterBuffer( &m_field_resource,
                                  m_field_buf,
                                  cudaGraphicsRegisterFlagsReadOnly );

    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }
}

FieldGLBufferUChar::~FieldGLBufferUChar()
{
    glDeleteTextures( 1, &m_field_gl_tex );
    if( m_field_resource != NULL ) {
        cudaGraphicsUnregisterResource( m_field_resource );
        m_field_resource = NULL;
    }
}

const unsigned char*
FieldGLBufferUChar::mapFieldBuffer( cudaStream_t stream )
{
    cudaError_t error;
    if( m_mapped ) {
        throw std::runtime_error( "buffer already mapped" );
    }
    error = cudaGraphicsMapResources( 1, &m_field_resource, stream );
    m_mapped = true;
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }
    unsigned char* ptr = NULL;
    size_t ptr_size = 0;
    error = cudaGraphicsResourceGetMappedPointer( (void**)&ptr, &ptr_size, m_field_resource );
    if( error != cudaSuccess ) {
        throw CUDAErrorException( error );
    }
    return ptr;
}

void
FieldGLBufferUChar::unmapFieldBuffer( cudaStream_t stream )
{
    if( !m_mapped ) {
        throw std::runtime_error( "buffer not mapped" );
    }
    cudaGraphicsUnmapResources( 1, &m_field_resource, stream );
    m_mapped = false;
}


} // of namespace cuhpmc
