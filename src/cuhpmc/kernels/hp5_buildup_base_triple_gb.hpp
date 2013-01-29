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

#include <cuda.h>
#include <builtin_types.h>

namespace cuhpmc {

void
run_hp5_buildup_base_triple_gb_ub( uint4*               hp_c_d,
                                   uint*                sb_c_d,
                                   const uint           hp2_N,
                                   uint4*               hp_b_d,
                                   uint4*               hp_a_d,
                                   unsigned char*       case_d,
                                   const float          iso,
                                   const uint3          chunks,
                                   const unsigned char* field,
                                   const uint3          field_size,
                                   const unsigned char* case_vtxcnt,
                                   cudaStream_t         stream );

} // of namespace cuhpmc
