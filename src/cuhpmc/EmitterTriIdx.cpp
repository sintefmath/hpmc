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
#include <vector_functions.h>
#include <iostream>
#include <cuhpmc/EmitterTriIdx.hpp>
#include <cuhpmc/Field.hpp>
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/FieldGlobalMemUChar.hpp>
#include <cuhpmc/IsoSurfaceIndexed.hpp>
#include "kernels/hp5_writer.hpp"

namespace cuhpmc {

EmitterTriIdx::EmitterTriIdx(IsoSurfaceIndexed *iso_surface )
    : m_constants( iso_surface->constants() ),
      m_field( iso_surface->field() ),
      m_iso_surface( iso_surface )
{
}

EmitterTriIdx::~EmitterTriIdx()
{
}


void
EmitterTriIdx::writeTriangleIndices( float* interleaved_buffer_d, uint triangles, cudaStream_t stream  )
{
    invokeKernel( interleaved_buffer_d, triangles, stream );
}


} // of namespace cuhpmc
