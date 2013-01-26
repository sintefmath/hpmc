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
#include <glhpmc/glhpmc.hpp>

namespace glhpmc {

enum HPMCVolumeLayout {
    HPMC_VOLUME_LAYOUT_CUSTOM,
    HPMC_VOLUME_LAYOUT_TEXTURE_3D
};

/** Specifies the layout of the scalar field. */
class Field
{
public:
    class Context {
        friend class Field;
    protected:
        GLint   m_loc_tex;
    };



    Field( HPMCConstants* constants );

    /** Invoked when IsoSurface is untainted.
     *
     * \returns true if everything is ok.
     */
    bool
    configure();

    const std::string
    fetcherSource( bool gradient ) const;

    /** configures a program that uses fetcher source. */
    bool
    setupProgram( Context& context, GLuint program ) const;

    /** binds textures and updates fetcher uniform values. */
    bool
    bind(const Context& context , GLuint texture_unit) const;

    GLsizei sizeX() const { return m_size[0]; }
    GLsizei sizeY() const { return m_size[1]; }
    GLsizei sizeZ() const { return m_size[2]; }

    GLsizei cellsX() const { return m_cells[0]; }
    GLsizei cellsY() const { return m_cells[1]; }
    GLsizei cellsZ() const { return m_cells[2]; }

    GLfloat extentX() const { return m_extent[0]; }
    GLfloat extentY() const { return m_extent[1]; }
    GLfloat extentZ() const { return m_extent[2]; }

    bool
    hasGradient() const { return m_tex_gradient_channels != GL_NONE; }

    bool
    isBinary() const { return m_binary; }

    /** The x,y,z-size of the lattice of scalar field samples. */
    GLsizei       m_size[3];
    /** The x,y,z-size of the MC grid, defaults to m_size-[1,1,1]. */
    GLsizei       m_cells[3];
    /** The extent of the MC grid when outputted from the traversal shader. */
    GLfloat       m_extent[3];


    /** Flag to denote if scalar field is binary (or continuous). */
    bool                   m_binary;

    HPMCVolumeLayout  m_mode;
    /** The source code of the custom fetch shader function (if custom fetch). */
    std::string       m_shader_source;
    /** The texture name of the Texture3D to fetch from (if fetch from Texture3D). */
    GLuint            m_tex;
    /** Channel that contains the scalar field. */
    GLenum            m_tex_field_channel;
    /** Channels that contain the gradient, or GL_NONE. */
    GLenum            m_tex_gradient_channels;

protected:
    HPMCConstants*  m_constants;

};

} // of namespace glhpmc
