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
#include <glhpmc/glhpmc_internal.hpp>
#include <glhpmc/Constants.hpp>
#include <glhpmc/IsoSurface.hpp>
#include <glhpmc/Logger.hpp>

namespace glhpmc {
static const std::string package = "HPMC.IsoSurface";

HPMCIsoSurface::HPMCIsoSurface( HPMCConstants* constants )
    : m_field( constants ),
      m_base_builder( this ),
      m_histopyramid( constants )
{
    m_tainted = true;
    m_broken = true;
    m_constants = constants;
    m_hp_build.m_tex_unit_1 = 0;
    m_hp_build.m_tex_unit_2 = 1;
}

bool
HPMCIsoSurface::init()
{
    if( !m_histopyramid.init() ) {
        return false;
    }

    return true;
}

void
HPMCIsoSurface::setAsBroken()
{
    m_broken = true;
}

void
HPMCIsoSurface::taint()
{
    m_tainted = true;
    m_broken = false;
}

bool
HPMCIsoSurface::untaint()
{
    if( !m_tainted ) {
        return true;
    }
    Logger log( m_constants, package + ".untaint" );

    bool retval = false;

    if(!m_field.configure() ) {
        log.errorMessage( "Failed to configure field" );
        return false;
    }

    if( !m_base_builder.configure() ) {
        log.errorMessage( "Failed to configure base level builder.");
        return false;
    }

    if( !m_histopyramid.configure( m_base_builder.log2Size() ) ) {
        log.errorMessage( "Failed to configure the HistoPyramid" );
        return false;
    }

//    else if ( !HPMCbuildHPBuildShaders( this ) ) {
//        HPMCLOG_ERROR( log, "Failed to build shaders." );
//    }

    m_tainted = false;
    retval = true;
    return retval;
}

GLsizei
HPMCIsoSurface::vertexCount()
{
    if( isBroken() ) {
        return 0;
    }
    return m_histopyramid.count();
}


void
HPMCIsoSurface::build()
{
    Logger log( m_constants, package + ".build" );
    if( isBroken() ) {
        log.errorMessage( "Invoked while broken" );
        return;
    }
    if( !untaint() ) {
        setAsBroken();
    }
    else {
        if( !m_base_builder.build( m_hp_build.m_tex_unit_1, m_hp_build.m_tex_unit_2 ) ) {
            setAsBroken();
        }
        else {
            if( !m_histopyramid.build( m_hp_build.m_tex_unit_1 ) ) {
                setAsBroken();
            }
        }
    }
}


HPMCIsoSurface::~HPMCIsoSurface()
{

}

} // of namespace glhpmc
