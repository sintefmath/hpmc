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
#include <vector>
#include <cmath>
#include <glhpmc/glhpmc_internal.hpp>
#include <sstream>
#include <stdexcept>

#include <glhpmc/Constants.hpp>
#include <glhpmc/Logger.hpp>

namespace glhpmc {
    static const std::string package = "hpmc.constants";



HPMCConstants::HPMCConstants( HPMCTarget target, HPMCDebugBehaviour debug )
    : m_target( target ),
      m_debug( debug ),
      m_vertex_counts( this),
      m_gpgpu_quad( this ),
      m_sequence_renderer( this ),
      m_edge_table( this )
{
    GLint gl_major, gl_minor;
    glGetIntegerv( GL_MAJOR_VERSION, &gl_major );
    glGetIntegerv( GL_MINOR_VERSION, &gl_minor );


    switch( target ) {
    case HPMC_TARGET_GL20_GLSL110:
        if( gl_major < 2 ) {
            std::cerr << "HPMC target is OpenGL 2.0, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 110\n";
        break;
    case HPMC_TARGET_GL21_GLSL120:
        if( (gl_major < 2) || ( (gl_major==2) && (gl_minor < 1) ) ) {
            std::cerr << "HPMC target is OpenGL 2.1, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 120\n";
        break;
    case HPMC_TARGET_GL30_GLSL130:
        if( gl_major < 3 ) {
            std::cerr << "HPMC target is OpenGL 3.0, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 130\n";
        break;
    case HPMC_TARGET_GL31_GLSL140:
        if( (gl_major < 3) || ( (gl_major==3) && (gl_minor < 1) ) ) {
            std::cerr << "HPMC target is OpenGL 3.1, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 140\n";
        break;
    case HPMC_TARGET_GL32_GLSL150:
        if( (gl_major < 3) || ( (gl_major==3) && (gl_minor < 2) ) ) {
            std::cerr << "HPMC target is OpenGL 3.2, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 150\n";
        break;
    case HPMC_TARGET_GL33_GLSL330:
        if( (gl_major < 3) || ( (gl_major==3) && (gl_minor < 3) ) ) {
            std::cerr << "HPMC target is OpenGL 3.3, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 330\n";
        break;
    case HPMC_TARGET_GL40_GLSL400:
        if( gl_major < 4 ) {
            std::cerr << "HPMC target is OpenGL 4.0, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 400\n";
        break;
    case HPMC_TARGET_GL41_GLSL410:
        if( (gl_major < 4) || ( (gl_major==4) && (gl_minor < 1) ) ) {
            std::cerr << "HPMC target is OpenGL 4.1, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 410\n";
        break;
    case HPMC_TARGET_GL42_GLSL420:
        if( (gl_major < 4) || ( (gl_major==4) && (gl_minor < 2) ) ) {
            std::cerr << "HPMC target is OpenGL 4.2, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 420\n";
        break;
    case HPMC_TARGET_GL43_GLSL430:
        if( (gl_major < 4) || ( (gl_major==4) && (gl_minor < 3) ) ) {
            std::cerr << "HPMC target is OpenGL 4.3, driver reports " << gl_major << "." << gl_minor << "." << std::endl;
            throw std::runtime_error( "Insufficient driver version" );
        }
        m_version_string = "#version 430\n";
        break;
    }

    if( target < HPMC_TARGET_GL30_GLSL130 ) {
        if( !GLEW_EXT_framebuffer_object ) {
            std::cerr << "HPMC target < OpenGL 3.0 requires GL_EXT_framebuffer_object extension, missing." << std::endl;
            throw std::runtime_error( "Missing extension" );
        }
        if( !GLEW_ARB_color_buffer_float ) {
            std::cerr << "HPMC target < OpenGL 3.0 requires GL_ARB_color_buffer_float extension, missing." << std::endl;
            throw std::runtime_error( "Missing extension" );
        }
    }
    Logger log( this, package + ".constructor", true );
    m_vertex_counts.init();
    m_gpgpu_quad.init();
    m_sequence_renderer.init();
    m_edge_table.init();
}


HPMCConstants::~HPMCConstants()
{
}

} // of namespace glhpmc
