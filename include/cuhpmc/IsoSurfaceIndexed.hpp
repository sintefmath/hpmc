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
#include <vector>
#include <cuhpmc/cuhpmc.hpp>
#include <cuhpmc/NonCopyable.hpp>
#include <cuhpmc/IsoSurface.hpp>

namespace cuhpmc {

class IsoSurfaceIndexed : public NonCopyable
{
public:
    IsoSurfaceIndexed( Field* field );

    virtual
    ~IsoSurfaceIndexed( );

    void
    build( float iso, cudaStream_t stream );

    Constants*
    constants() { return m_constants; }

    Field*
    field() { return m_field; }

    /** Returns the size of the hp5 histopyramid. */
    uint
    hp5Size() const { return m_hp5_size; }

    uint
    hp5Levels() const { return m_hp5_levels; }

    const uint3
    hp5Chunks() const { return m_hp5_chunks; }

    const std::vector<uint>&
    hp5Offsets() const { return m_hp5_offsets; }

    uint
    triangles();

    uint
    vertices();

    /** Return the ISO value for which this surface was most recently built. */
    float
    iso() const { return m_iso; }

    /** Returns a device pointer to the hp5 histopyramid data. */
    const uint4*
    trianglePyramidDev() const { return m_triangle_pyramid_d; }

    const uint4*
    vertexPyramidDev() const { return m_vertex_pyramid_d; }

    /** Returns a device pointer to an array of hp5 level offsets. */
    const uint*
    hp5LevelOffsetsDev() const { return m_hp5_offsets_d; }

    const unsigned char*
    mcCasesDev() const { return m_case_d; }

protected:
    Constants*          m_constants;
    Field*      m_field;
    uint3               m_cells;
    float               m_iso;
    uint3               m_hp5_chunks;
    uint                m_hp5_input_N;
    uint                m_hp5_levels;
    uint                m_hp5_first_single_level;
    uint                m_hp5_first_double_level;
    uint                m_hp5_first_triple_level;
    uint                m_hp5_size;
    std::vector<uint>   m_hp5_level_sizes;
    std::vector<uint>   m_hp5_offsets;


    cudaEvent_t         m_buildup_event;

    uint*               m_vertex_triangle_top_h;    // populated using zero-copy
    uint*               m_vertex_triangle_top_d;

    uint*               m_hp5_offsets_d;
    uint4*              m_triangle_pyramid_d;
    uint4*              m_vertex_pyramid_d;
    uint*               m_triangle_sideband_d;     // sideband buffer
    uint*               m_vertex_sideband_d;     // sideband buffer
    unsigned char*      m_case_d;

    void
    invokeBaseBuildup( cudaStream_t stream );

    void
    invokeDoubleBuildup( uint level_a, cudaStream_t stream );

    void
    invokeSingleBuildup( uint level_a, cudaStream_t stream );

    void
    invokeApexBuildup( cudaStream_t stream );

};


} // of namespace cuhpmc
