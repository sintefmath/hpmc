#pragma once
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
#include <cuhpmc/cuhpmc.hpp>
#include <cuhpmc/Field.hpp>

namespace cuhpmc {

class FieldGLBufferUChar : public Field
{
public:
    FieldGLBufferUChar( Constants*     constants,
                        GLuint         field_buf,
                        uint           width,
                        uint           height,
                        uint           depth );

    ~FieldGLBufferUChar();

    GLuint
    fieldGLTex() const { return m_field_gl_tex; }

    const unsigned char*
    mapFieldBuffer( cudaStream_t stream );

    void
    unmapFieldBuffer( cudaStream_t stream );


protected:
    bool                    m_mapped;
    GLuint                  m_field_buf;
    GLuint                  m_field_gl_tex;
    cudaGraphicsResource*   m_field_resource;

};

} // of namespace cuhpmc
