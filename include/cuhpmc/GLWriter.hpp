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
#include <iosfwd>
#include <string>
#include <cuhpmc/cuhpmc.hpp>
#include <cuhpmc/AbstractWriter.hpp>

namespace cuhpmc {

class GLWriter : public AbstractWriter
{
public:
    GLWriter( GLIsoSurface* iso_surface );

    ~GLWriter();

    void
    render( const GLfloat* modelview_projection,
            const GLfloat* normal_matrix,
            cudaStream_t stream );

protected:
    bool    m_conf_constmem_apex;
    GLuint  m_program;
    GLint   m_loc_iso;
    GLint   m_loc_hp_offsets;
    GLint   m_loc_mvp;
    GLint   m_loc_nm;
    GLint   m_block_ix_apex;

    /** Adds line numbers to a source string and writes it to out. */
    void
    dumpSource( std::stringstream& out, const std::string& source );

    /** Creates and compules a shader, writes compiler output to out. */
    GLuint
    compileShader( std::stringstream& out, const std::string& src, GLenum type ) const;

    /** Links a shader program, writes linker output to out. */
    bool
    linkShaderProgram( std::stringstream& out, GLuint program ) const;

};



} // of namespace cuhpmc
