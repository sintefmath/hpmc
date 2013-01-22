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
#include <cuhpmc/AbstractIsoSurface.hpp>

namespace cuhpmc {

class IsoSurface : public AbstractIsoSurface
{
public:
    IsoSurface( AbstractField* field, cudaStream_t stream = 0 );

    ~IsoSurface( );

    cudaStream_t
    stream() const { return m_stream; }

    void
    build( float iso );

    size_t
    vertices();


protected:
    cudaStream_t        m_stream;
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
    uint4*              m_hp5_hp_d;
    uint*               m_hp5_sb_d;
    unsigned char*      m_case_d;

    uint*               m_hp5_top_h;    // populated using zero-copy
    uint*               m_hp5_top_d;

};


} // of namespace cuhpmc
