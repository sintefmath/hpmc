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
run_dummy_writer( float*                output_d,
                  const uint4*          hp5_pyramid_d,
                  const unsigned char*  cases_d,
                  const uint*           hp5_level_offsets_d,
                  const uint3           hp5_chunks,
                  const uint            hp5_size,
                  const uint            hp5_max_level,
                  const uint            triangles,
                  const unsigned char*  field,
                  const uint3           field_size,
                  cudaStream_t          stream );

} // of namespace cuhpmc
