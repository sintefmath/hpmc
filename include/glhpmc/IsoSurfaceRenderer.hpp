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

#include <string>
#include <glhpmc/glhpmc.hpp>
#include <glhpmc/Field.hpp>

namespace glhpmc {

struct HPMCIsoSurfaceRenderer
{
public:
    HPMCIsoSurfaceRenderer( struct HPMCIsoSurface* iso_surface );

    ~HPMCIsoSurfaceRenderer();

    const std::string
    extractionSource() const;

    bool
    setProgram( GLuint program,
                GLuint tex_unit_work1,
                GLuint tex_unit_work2,
                GLuint tex_unit_work3 );

    bool
    draw( int transform_feedback_mode, bool flip_orientation );

    HPMCIsoSurface*  m_handle;
    GLuint                    m_program;
    GLuint                    m_scalarfield_unit;
    GLuint                    m_histopyramid_unit;
    GLuint                    m_edge_decode_unit;
    GLint                     m_offset_loc;
    GLint                     m_threshold_loc;

protected:
    Field::ProgramContext*  m_field_context;


};

} // of namespace glhpmc
