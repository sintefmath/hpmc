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

namespace cuhpmc {

class AbstractIsoSurface : public NonCopyable
{
public:

    virtual
    ~AbstractIsoSurface( );

    Constants*
    constants() { return m_constants; }

    AbstractField*
    field() { return m_field; }

    /** Returns the size of the hp5 histopyramid. */
    uint
    hp5Size() const { return m_hp5_size; }

    uint
    hp5Levels() const { return m_hp5_levels; }

    const uint3
    hp5Chunks() const { return m_hp5_chunks; }

    uint
    triangles();

protected:
    Constants*          m_constants;
    AbstractField*      m_field;
    uint3               m_cells;
    uint3               m_hp5_chunks;
    uint                m_hp5_input_N;
    uint                m_hp5_levels;
    uint                m_hp5_first_single_level;
    uint                m_hp5_first_double_level;
    uint                m_hp5_first_triple_level;
    uint                m_hp5_size;
    std::vector<uint>   m_hp5_level_sizes;
    std::vector<uint>   m_hp5_offsets;

    uint*               m_hp5_sb_d;     // sideband buffer

    cudaEvent_t         m_buildup_event;

    uint*               m_hp5_top_h;    // populated using zero-copy
    uint*               m_hp5_top_d;

    AbstractIsoSurface( AbstractField* field );

    void
    buildNonIndexed( float iso, uint4* hp5_hp_d, unsigned char* case_d, cudaStream_t stream );

};


} // of namespace cuhpmc
