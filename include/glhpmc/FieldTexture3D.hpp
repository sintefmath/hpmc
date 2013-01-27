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
#include <glhpmc/Field.hpp>

namespace glhpmc {

/** Field for sampling an existing OpenGL 3D texture.
 *
 * If gradients is true, the field is assumed to have four components, where the
 * rgb-channels encode the gradient while the a-channel encode the scalar field.
 *
 * If gradients is false, gradients are found on the fly using forward
 * differences. If the target is pre-OpenGL 3.0, the field is assumed to reside
 * in the alpha-channel, while for targets from OpenGL 3.0 and up, the field is
 * assumed to reside in the red channel. The reason for this is compatibility,
 * early OpenGL 2.0 vertex shaders could only fetch from either float RGBA or A
 * textures in the vertex shaders, while with OpenGL 3.0, single channeled
 * textures should reside in the red channel.
 *
 * It is the developer's responsibility that the texture exists and maintains
 * its dimensions throughout the lifetime of this object. However, the actual
 * contents may be changed dynamically.
 *
 */
class FieldTexture3D : public Field
{
public:

    FieldTexture3D( glhpmc::HPMCConstants* constants,
                    GLuint                 sample_unit,
                    GLuint                 texture,
                    bool                   gradients,
                    GLsizei                samples_x,
                    GLsizei                samples_y,
                    GLsizei                samples_z );


    Field::ProgramContext*
    createContext( GLuint program ) const;

    bool
    gradients() const;

    const std::string
    fetcherFieldSource( ) const;

    const std::string
    fetcherFieldAndGradientSource( ) const;

    bool
    bind( ProgramContext* program_context ) const;

    bool
    unbind( ProgramContext* program_context ) const;


protected:
    GLint   m_sampler;
    GLint   m_texture;
    bool    m_gradients;

};

} // of namespace glhpmc
