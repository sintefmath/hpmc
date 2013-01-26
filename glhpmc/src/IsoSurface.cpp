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
#include <sstream>
#include <glhpmc/glhpmc_internal.hpp>
#include <glhpmc/Constants.hpp>
#include <glhpmc/IsoSurface.hpp>
#include <glhpmc/Logger.hpp>

namespace glhpmc {
static const std::string package = "HPMC.IsoSurface";

HPMCIsoSurface*
HPMCIsoSurface::factory( HPMCConstants* constants )
{
    if( constants == NULL ) {
        return NULL;
    }
    Logger log( constants, package + ".factory", true );
    try {
        HPMCIsoSurface* h = new HPMCIsoSurface( constants );
        h->init();
        return h;
    }
    catch( std::runtime_error& e ) {
        log.errorMessage( e.what() );
    }
    return NULL;
}


void
HPMCIsoSurface::setLatticeSize( GLsizei x_size, GLsizei y_size, GLsizei z_size )
{
    Logger log( m_constants, package + ".setLatticeSize", true );

    m_field.m_size[0] = x_size;
    m_field.m_size[1] = y_size;
    m_field.m_size[2] = z_size;
    m_field.m_cells[0] = std::max( (GLsizei)1u, m_field.m_size[0] )-(GLsizei)1u;
    m_field.m_cells[1] = std::max( (GLsizei)1u, m_field.m_size[1] )-(GLsizei)1u;
    m_field.m_cells[2] = std::max( (GLsizei)1u, m_field.m_size[2] )-(GLsizei)1u;
    taint();

    if( m_constants->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "field.size = [ "
          << m_field.m_size[0] << " x "
          << m_field.m_size[1] << " x "
          << m_field.m_size[2] << " ]";
        log.debugMessage( o.str() );

        o.str("");
        o << "field.cells = [ "
          << m_field.m_cells[0] << " x "
          << m_field.m_cells[1] << " x "
          << m_field.m_cells[2] << " ]";
        log.debugMessage( o.str() );
    }
}


void
HPMCIsoSurface::setGridSize( GLsizei x_size, GLsizei y_size, GLsizei z_size )
{
    Logger log( m_constants, package + ".setGridSize", true );
    m_field.m_cells[0] = x_size;
    m_field.m_cells[1] = y_size;
    m_field.m_cells[2] = z_size;
    taint();
    if( m_constants->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "field.cells = [ "
          << m_field.m_cells[0] << " x "
          << m_field.m_cells[1] << " x "
          << m_field.m_cells[2] << " ]";
        log.debugMessage( o.str() );
    }
}


void
HPMCIsoSurface::setGridExtent( GLsizei x_extent, GLsizei y_extent, GLsizei z_extent )
{
    Logger log( m_constants, package + ".setGridExtent", true );
    m_field.m_extent[0] = x_extent;
    m_field.m_extent[1] = y_extent;
    m_field.m_extent[2] = z_extent;
    taint();
    if( m_constants->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "grid.extent = [ "
          << m_field.m_extent[0] << " x "
          << m_field.m_extent[1] << " x "
          << m_field.m_extent[2] << " ]";
        log.debugMessage( o.str() );
    }
}


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
