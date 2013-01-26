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
#include <string>
#include <iostream>
#include <glhpmc/glhpmc.hpp>

namespace glhpmc {

enum HPMCVolumeLayout {
    HPMC_VOLUME_LAYOUT_CUSTOM,
    HPMC_VOLUME_LAYOUT_TEXTURE_3D
};

/** Abstract base class for field. */
class Field
{
public:
    struct ProgramContext
    {
        virtual
        ~ProgramContext() {}
    };

    virtual
    ~Field() {}

    /** can assume that program is bound. */
    virtual
    ProgramContext*
    createContext( GLuint program ) const { return NULL; }

    virtual
    bool
    gradients() const = 0;

    virtual
    const std::string
    fetcherSource( bool gradient ) const = 0;

    virtual
    bool
    bind( ProgramContext* program_context ) const { return true; }

    virtual
    bool
    unbind( ProgramContext* program_context ) const { return true; }

    unsigned int
    samplesX() const { return m_samples_x; }

    unsigned int
    samplesY() const { return m_samples_y; }

    unsigned int
    samplesZ() const { return m_samples_z; }


protected:
    Field( HPMCConstants* constants,
           unsigned int samples_x,
           unsigned int samples_y,
           unsigned int samples_z )
        : m_constants( constants ),
          m_samples_x( samples_x ),
          m_samples_y( samples_y ),
          m_samples_z( samples_z )
    {
    }

    HPMCConstants*
    constants() { return m_constants; }

    const HPMCConstants*
    constants() const { return m_constants; }

private:
    HPMCConstants*  m_constants;
    unsigned int    m_samples_x;
    unsigned int    m_samples_y;
    unsigned int    m_samples_z;

};




} // of namespace glhpmc
