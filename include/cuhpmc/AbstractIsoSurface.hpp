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

class AbstractIsoSurface : public NonCopyable
{
public:

    virtual
    ~AbstractIsoSurface( );

    Constants*
    constants() { return m_constants; }

    AbstractField*
    field() { return m_field; }


protected:
    Constants*      m_constants;
    AbstractField*  m_field;

    AbstractIsoSurface( AbstractField* field );

};


} // of namespace cuhpmc
