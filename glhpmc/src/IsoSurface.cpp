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


HPMCIsoSurface::HPMCIsoSurface( HPMCConstants* constants,
                                Field* field,
                                unsigned int cells_x,
                                unsigned int cells_y,
                                unsigned int cells_z )
    : m_tainted( true ),
      m_broken( false ),
      m_constants( constants ),
      m_field( field ),
      m_cells_x( cells_x ),
      m_cells_y( cells_y ),
      m_cells_z( cells_z ),
      m_base_builder( this ),
      m_histopyramid( constants )
{
    m_hp_build.m_tex_unit_1 = 0;
    m_hp_build.m_tex_unit_2 = 1;
    if( constants == NULL ) {
        throw std::runtime_error( "constants == NULL" );
    }
    if( field == NULL ) {
        throw std::runtime_error( "field == NULL" );
    }
    if( (m_cells_x == 0) || (m_cells_x >= m_field->samplesX()) ) {
        m_cells_x = m_field->samplesX()-1;
    }
    if( (cells_y == 0) || (m_cells_y >= m_field->samplesY()) ) {
        m_cells_y = m_field->samplesY()-1;
    }
    if( (cells_z == 0) || (m_cells_z >= m_field->samplesZ()) ) {
        m_cells_z = m_field->samplesZ()-1;
    }
    Logger log( constants, package + ".constructor", true );
    m_histopyramid.init();
}

HPMCIsoSurface::~HPMCIsoSurface()
{
}

void
HPMCIsoSurface::build( GLfloat iso )
{
    Logger log( m_constants, package + ".build", true );
    if( m_broken ) {
        log.errorMessage( "Invoked while broken" );
        return;
    }

    // --- store state ---------------------------------------------------------
    GLint old_viewport[4];
    glGetIntegerv( GL_VIEWPORT, old_viewport );

    // set iso
    m_threshold = iso;
    // try to untaint
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
    // --- restore state -------------------------------------------------------
    if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
        glBindVertexArray( 0 ); // GPGPU quad uses VAO.
    }
    glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glViewport( old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3] );
    glUseProgram( 0 );

    // should check for errors and h->setAsBroken()?

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

//    if(!m_old_field.configure() ) {
//        log.errorMessage( "Failed to configure field" );
//        return false;
//    }

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





} // of namespace glhpmc
