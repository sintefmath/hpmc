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
#include <cuhpmc/cuhpmc.hpp>
#include <cuhpmc/NonCopyable.hpp>

namespace cuhpmc {

class EmitterTriIdx : public NonCopyable
{
public:
    EmitterTriIdx( IsoSurfaceIndexed* iso_surface );

    virtual
    ~EmitterTriIdx();

    void
    writeTriangleIndices( float* interleaved_buffer_d, uint triangles, cudaStream_t stream  );

protected:
    Constants*          m_constants;
    Field*              m_field;
    IsoSurfaceIndexed*  m_iso_surface;


};


} // of namespace cuhpmc
