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
#include <stdexcept>
#include <cuhpmc/AbstractField.hpp>

namespace cuhpmc {


AbstractField::AbstractField(  Constants* constants, uint width, uint height, uint depth )
    : m_constants( constants ),
      m_width( width ),
      m_height( height ),
      m_depth( depth )
{
    if( m_constants == NULL ) {
        throw std::runtime_error( "constants == NULL" );
    }
    if( m_width < 4 ) {
        throw std::runtime_error( "width < 4" );
    }
    if( m_height < 4 ) {
        throw std::runtime_error( "height < 4" );
    }
    if( m_depth < 4 ) {
        throw std::runtime_error( "depth < 4" );
    }
}

Constants*
AbstractField::constants()
{
    return m_constants;
}

AbstractField::~AbstractField()
{
}

} // of namespace cuhpmc
