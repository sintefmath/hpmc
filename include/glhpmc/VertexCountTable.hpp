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
#include <glhpmc/glhpmc.hpp>

namespace glhpmc {

/** Table containing the number of vertices needed for each Marching Cubes case. */
class HPMCVertexCountTable
{
public:
    HPMCVertexCountTable( HPMCConstants* constants );

    ~HPMCVertexCountTable( );

    /** Initializes object.
     *
     * \sideeffect active texture,
     *             texture binding
     */
    void
    init( );

    GLuint
    texture() const { return m_texture; }

protected:
    HPMCConstants*  m_constants;
    GLuint          m_texture;
};

} // of namespace glhpmc
