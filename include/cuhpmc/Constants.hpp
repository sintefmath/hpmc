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
#include <cuhpmc/NonCopyable.hpp>

namespace cuhpmc {

/** Per device constants. */
class Constants : public NonCopyable
{
public:
    Constants();

    ~Constants();

    /** Returns the number of unique vertices in upper nibble and the number of triangles in the lower nibble for each MC case. */
    const unsigned char*
    vertexTriangleCountDev() const { return m_vertex_triangle_count_dev; }

    /** Returns the number of triangles produced for each MC case. */
    const unsigned char*
    triangleCountDev() const { return m_triangle_count_dev; }

    const unsigned char*
    caseIntersectEdgeDev() const { return m_case_intersect_edge_d; }

    const unsigned char*
    caseIndexIntersectEdgeDev() const { return m_case_indexed_intersect_edge_d; }

    GLuint
    caseIntersectEdgeGL() { return m_case_intersect_edge_tex; }

private:
    unsigned char*  m_vertex_triangle_count_dev;
    unsigned char*  m_triangle_count_dev;
    unsigned char*  m_case_intersect_edge_d;
    unsigned char*  m_case_indexed_intersect_edge_d;
    GLuint          m_case_intersect_edge_tex;


};


} // of namespace cuhpmc
