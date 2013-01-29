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
#include <cuhpmc/IsoSurface.hpp>

namespace cuhpmc {

/** Iso-surface that keeps the buffers needed for direct renderering in OpenGL.
 */
class IsoSurfaceGL : public IsoSurface
{
public:
    IsoSurfaceGL( Field* field );

    ~IsoSurfaceGL( );

    void
    build( float iso, cudaStream_t stream );



    GLuint
    caseBuf() const { return m_case_buf; }

    GLuint
    caseGLTex() const { return m_case_gl_tex; }

    GLuint
    hp5GLBuf() const { return m_hp5_hp_buf; }

    GLuint
    hp5GLTex() const { return m_hp5_gl_tex; }

protected:
    GLuint                  m_hp5_hp_buf;
    GLuint                  m_case_buf;
    GLuint                  m_hp5_gl_tex;
    GLuint                  m_case_gl_tex;

    cudaGraphicsResource*   m_resources[2];
};


} // of namespace cuhpmc
