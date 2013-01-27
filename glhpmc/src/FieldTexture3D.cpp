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
#include <sstream>
#include <glhpmc/Constants.hpp>
#include <glhpmc/FieldTexture3D.hpp>

namespace glhpmc {


FieldTexture3D::FieldTexture3D( glhpmc::HPMCConstants* constants,
                                GLuint                 sample_unit,
                                GLuint                 texture,
                                bool                   gradients,
                                GLsizei                samples_x,
                                GLsizei                samples_y,
                                GLsizei                samples_z )
    : Field( constants, samples_x, samples_y, samples_z ),
      m_sampler( sample_unit ),
      m_texture( texture ),
      m_gradients( gradients )
{
}


Field::ProgramContext*
FieldTexture3D::createContext( GLuint program ) const
{
    GLint loc = glGetUniformLocation( program, "HPMC_scalarfield" );
    std::cerr << "loc=" << loc << "\n";
    glUniform1i( loc, m_sampler );
    return NULL;
}

bool
FieldTexture3D::gradients() const
{
    return m_gradients;
}

const std::string
FieldTexture3D::fetcherFieldSource( ) const
{
    std::stringstream o;

    o << "uniform sampler3D HPMC_scalarfield;\n";
    o << "float\n";
    o << "HPMC_fetch( vec3 p )\n";
    o << "{\n";
    if( constants()->target() < HPMC_TARGET_GL30_GLSL130 ) {
        o << "    return texture3D( HPMC_scalarfield, p ).a;\n";
    }
    else {
        if( m_gradients ) {
            o << "    return texture( HPMC_scalarfield, p ).a;\n";
        }
        else {
            o << "    return texture( HPMC_scalarfield, p ).r;\n";
        }
    }
    o << "}\n";
    return o.str();
}

const std::string
FieldTexture3D::fetcherFieldAndGradientSource() const
{
    std::stringstream o;

    o << "uniform sampler3D HPMC_scalarfield;\n";
    o << "vec4\n";
    o << "HPMC_fetchGrad( vec3 p )\n";
    o << "{\n";
    if( constants()->target() < HPMC_TARGET_GL30_GLSL130 ) {
        o << "    return texture3D( HPMC_scalarfield, p );\n";
    }
    else {
        o << "    return texture( HPMC_scalarfield, p );\n";
    }
    o << "}\n";
    return o.str();
}

bool
FieldTexture3D::bind( ProgramContext* program_context ) const
{
    glActiveTexture( GL_TEXTURE0 + m_sampler );
    glBindTexture( GL_TEXTURE_3D, m_texture );
    return true;
}

bool
FieldTexture3D::unbind( ProgramContext* program_context ) const
{
    glActiveTexture( GL_TEXTURE0 + m_sampler );
    glBindTexture( GL_TEXTURE_3D, 0 );
    return true;
}


} // of namespace glhpmc
