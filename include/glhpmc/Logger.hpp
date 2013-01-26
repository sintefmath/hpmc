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
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
class HPMCConstants;

class Logger
{
public:
    Logger( const HPMCConstants* constants, const std::string& where, bool force_check = false );

    ~Logger();

    bool
    doDebug() const { return true; }

    void
    debugMessage( const std::string& msg );

    void
    warningMessage( const std::string& msg );

    void
    errorMessage( const std::string& msg );

    void
    setObjectLabel( GLenum identifier, GLuint name, const std::string label );

    const std::string&
    where() const { return m_where; }

protected:
    const HPMCConstants*    m_constants;
    const std::string       m_where;
    const bool              m_force_check;

    const std::string
    glErrorString( GLenum error ) const;

};

#endif // LOGGER_HPP
