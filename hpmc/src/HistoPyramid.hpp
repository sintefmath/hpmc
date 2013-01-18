/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
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
#ifndef HISTOPYRAMID_HPP
#define HISTOPYRAMID_HPP
#include <vector>
#include <hpmc.h>

class HPMCGPGPUQuad;

/** Information about the HistoPyramid texture. */
class HPMCHistoPyramid
{
public:
    class Context {
        friend class HPMCHistoPyramid;
    protected:
        GLint   m_loc_hp_tex;
    };

    HPMCHistoPyramid( HPMCConstants* constants );

    ~HPMCHistoPyramid();

    bool init();

    bool configure( GLsizei size_l2 );

    bool build( GLint tex_unit_a);

    GLsizei count();

    GLsizei size() const { return m_size; }

    GLuint texture() const { return m_tex; }

    GLuint baseFramebufferObject() const { return m_fbos[0]; }

protected:
    HPMCConstants*      m_constants;
    GLsizei             m_size;                 ///< Size of \ref m_tex.
    GLsizei             m_size_l2;              ///< Log2 size of \ref m_tex.
    GLuint              m_tex;                  ///< HP tex (tex is quadratic mipmapped RGBA32F TEXTURE_2D).
    std::vector<GLuint> m_fbos;                 ///< FBOs bound to each mipmap level of \ref m_tex.
    GLuint              m_top_pbo;              ///< PBO for async readback of \ref m_tex top element.
    GLsizei             m_top_count;            ///< CPU copy of \ref m_tex top element.
    GLsizei             m_top_count_updated;    ///< Tags that \m_top_pbo is more recent than \ref m_top_count.
    GLuint              m_reduce1_program;      ///< Shader program for first reduction.
    GLint               m_reduce1_loc_delta;    ///< Pixel deltas for first reduction (GL 2.x).
    GLint               m_reduce1_loc_level;    ///< Source mipmap level for first reduction (GL 3.x and up).
    GLint               m_reduce1_loc_hp_tex;   ///< Texture unit for \ref m_tex for first reduction.
    GLuint              m_reducen_program;      ///< Shader program for subsequent reductions.
    GLint               m_reducen_loc_delta;    ///< Pixel deltas for subsequent reduction (GL 2.x).
    GLint               m_reducen_loc_level;    ///< Source mipmap level for subsequent reduction (GL 3.x and up).
    GLint               m_reducen_loc_hp_tex;   ///< Texture unit for \ref m_tex for subsequent reductions.


};

#endif // HISTOPYRAMID_HPP
