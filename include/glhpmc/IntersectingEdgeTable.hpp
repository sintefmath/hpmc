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

/** Marching Cubes table with vertices encoded as edge intersections.
 *
 * Each vertex produced by the marching cubes is located on a cell edge. This
 * table encodes which cell edge that vertex is on.
 */
class HPMCIntersectingEdgeTable
{
public:
    HPMCIntersectingEdgeTable( HPMCConstants* constants );

    ~HPMCIntersectingEdgeTable();

    /** Initializes object.
     *
     * \sideeffect active texture,
     *             texture binding
     */
    void
    init( );

    /** Normal table. */
    GLuint
    texture() const { return m_tex; }

    /** Same as \ref texture, but with orientation of triangles flipped. */
    GLuint
    textureFlip() const { return m_tex_flip; }

    /** Same as \ref texture, but with normal vectors for binary fields encoded in fractional parts of x,y,z. */
    GLuint
    textureNormal() const { return m_tex_normal; }

    /** Same as \ref textureNormal, but with orientation of triangles flipped. */
    GLuint
    textureNormalFlip() const { return m_tex_normal_flip; }

protected:
    HPMCConstants*  m_constants;
    GLuint          m_tex;
    GLuint          m_tex_flip;
    GLuint          m_tex_normal;
    GLuint          m_tex_normal_flip;
};

} // of namespace glhpmc
