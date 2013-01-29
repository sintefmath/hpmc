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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <vector_functions.h>
#include <stdexcept>
#include <cuhpmc/IsoSurface.hpp>
#include <cuhpmc/Field.hpp>
#include <cuhpmc/FieldGlobalMemUChar.hpp>
#include <cuhpmc/Constants.hpp>

namespace cuhpmc {

IsoSurface::IsoSurface( Field* field )
    : AbstractIsoSurface( field )
{


    cudaMalloc( (void**)&m_hp5_hp_d, 4*sizeof(uint)* m_hp5_size );
    cudaMalloc( (void**)&m_case_d, m_hp5_input_N );

    cudaMalloc( (void**)&m_hp5_offsets_d, sizeof(uint)*32 );
    cudaMemset( m_hp5_offsets_d, 0, sizeof(uint)*32 );
    cudaMemcpy( m_hp5_offsets_d, m_hp5_offsets.data(), sizeof(uint)*m_hp5_offsets.size(), cudaMemcpyHostToDevice );

}

IsoSurface::~IsoSurface( )
{
}


void
IsoSurface::build( float iso, cudaStream_t stream )
{
    buildNonIndexed( iso, m_hp5_hp_d, m_case_d, stream );

    /*
    uint3 field_size = make_uint3( m_field->width(), m_field->height(), m_field->depth() );

    if( FieldGlobalMemUChar* field = dynamic_cast<FieldGlobalMemUChar*>( m_field ) ) {
        run_hp5_buildup_base_triple_gb_ub( m_hp5_hp_d + m_hp5_offsets[ m_hp5_levels-3 ],
                                           m_hp5_sb_d + m_hp5_offsets[ m_hp5_levels-3 ],
                                           m_hp5_level_sizes[ m_hp5_levels-1 ],
                                           m_hp5_hp_d + m_hp5_offsets[ m_hp5_levels-2 ],
                                           m_hp5_hp_d + m_hp5_offsets[ m_hp5_levels-1 ],
                                           m_case_d,
                                           iso,
                                           m_hp5_chunks,
                                           field->fieldDev(),
                                           field_size,
                                           m_constants->triangleIndexCountDev(),
                                           stream );


    }
    else {
        throw std::runtime_error( "Unsupported field type" );
    }
    for( uint i=m_hp5_first_triple_level; i>m_hp5_first_double_level; i-=2 ) {
        run_hp5_buildup_level_double( m_hp5_hp_d + m_hp5_offsets[i-2],
                                      m_hp5_sb_d + m_hp5_offsets[i-2],
                                      m_hp5_hp_d + m_hp5_offsets[i-1],
                                      m_hp5_sb_d + m_hp5_offsets[i],
                                      m_hp5_level_sizes[i-1],
                                      stream );
    }
    for( uint i=m_hp5_first_double_level; i>m_hp5_first_single_level; --i ) {
        run_hp5_buildup_level_single( m_hp5_hp_d + m_hp5_offsets[ i-1 ],
                                      m_hp5_sb_d + m_hp5_offsets[ i-1 ],
                                      m_hp5_sb_d + m_hp5_offsets[ i   ],
                                      m_hp5_level_sizes[i-1],
                                      stream );
    }
    run_hp5_buildup_apex( m_hp5_top_d,
                          m_hp5_hp_d,
                          m_hp5_sb_d + 32,
                          m_hp5_level_sizes[2],
                          stream );
    cudaEventRecord( m_buildup_event, stream );
*/
}




} // of namespace cuhpmc
