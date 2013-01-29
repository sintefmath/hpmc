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
#include <cuhpmc/IsoSurfaceCUDA.hpp>
#include <cuhpmc/FieldGlobalMemUChar.hpp>
#include <cuhpmc/EmitterTriVtxCUDA.hpp>
#include "kernels/hp5_writer.hpp"

namespace cuhpmc {

EmitterTriVtxCUDA::EmitterTriVtxCUDA( IsoSurface* iso_surface )
    : EmitterTriVtx( iso_surface )
{}

void
EmitterTriVtxCUDA::writeInterleavedNormalPosition( float* interleaved_buffer_d, uint triangles, cudaStream_t stream )
{
    if( IsoSurfaceCUDA* iso_surface = dynamic_cast<IsoSurfaceCUDA*>( m_iso_surface ) ) {

        if( FieldGlobalMemUChar* field = dynamic_cast<FieldGlobalMemUChar*>( m_field ) ) {

            run_dummy_writer( interleaved_buffer_d,
                              iso_surface->hp5Dev(),
                              iso_surface->hp5LevelOffsetsDev(),
                              iso_surface->hp5Chunks(),
                              iso_surface->hp5Size(),
                              iso_surface->hp5Levels(),
                              triangles,
                              field->fieldDev(),
                              make_uint3( field->width(),
                                          field->height(),
                                          field->depth() ),
                              stream );
        }
    }

}


} // of namespace cuhpmc
