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

class IsoSurface : public NonCopyable
{
public:

    virtual
    ~IsoSurface( );

    virtual
    void
    build( float iso, cudaStream_t stream ) = 0;

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

    /** Return the ISO value for which this surface was most recently built. */
    float
    iso() const { return m_iso; }

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

    uint*               m_hp5_sb_d;     // sideband buffer

    cudaEvent_t         m_buildup_event;

    uint*               m_hp5_top_h;    // populated using zero-copy
    uint*               m_hp5_top_d;

    IsoSurface( Field* field );

    void
    buildNonIndexed( float iso, uint4* hp5_hp_d, unsigned char* case_d, cudaStream_t stream );

    void
    invokeBaseBuildup( uint4*               hp_c_d,
                       uint*                sb_c_d,
                       const uint           hp2_N,
                       uint4*               hp_b_d,
                       uint4*               hp_a_d,
                       unsigned char*       case_d,
                       const float          iso,
                       const uint3          chunks,
                       const unsigned char* field,
                       const uint3          field_size,
                       const unsigned char *case_vtxcnt,
                       cudaStream_t         stream );

    /** Build levels b and c from sideband of a. */
    void
    invokeDoubleBuildup( uint4*         pyramid_c_d,
                         uint*          sideband_c_d,
                         uint4*         pyramid_b_d,
                         const uint*    sideband_a_d,
                         const uint     N_b,
                         cudaStream_t   stream );

    void
    invokeSingleBuildup( uint4*        hp_b_d,
                         uint*         sb_b_d,
                         const uint*   sb_a_d,
                         const uint    N_b,
                         cudaStream_t  stream );

    void
    invokeApexBuildup( uint*         sum_d,
                       uint4*        hp_dcb_d,
                       const uint*   sb_a_d,
                       const uint    N_a,
                       cudaStream_t  stream );

};


} // of namespace cuhpmc
