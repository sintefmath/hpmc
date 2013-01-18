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
#ifndef GPGPUQUAD_HPP
#define GPGPUQUAD_HPP

class HPMCConstants;

/** Helper class to render GPGPU quads. */
class HPMCGPGPUQuad
{
public:
    HPMCGPGPUQuad( HPMCConstants* constants );

    /** Initializes object.
     *
     * \sideeffect GL_ARRAY_BUFFER binding,
     *             vertex array buffer binding,
     */
    void
    init();

    ~HPMCGPGPUQuad( );

    /** Sets up vertex data for GPGPU quad rendering.
     *
     * \sideeffect GL_VERTEX_ARRAY (< glsl130)
     *             GL_VERTEX_ARRAY_SIZE (< glsl130),
     *             GL_VERTEX_ARRAY_TYPE (< glsl130),
     *             GL_VERTEX_ARRAY_STRIDE (< glsl130),
     *             GL_VERTEX_ARRAY_POINTER (< glsl130,
     *             GL_VERTEX_ARRAY_OBJECT (>= glsl130)
     */
    void
    bindVertexInputs() const;

    /** Renders a GPGPU quad. */
    void
    render() const;

    /** Returns shader object of a pass through vertex shader.
     *
     * If target is less than GL3.0, gl_TexCoord[0] is contains the [0,1]
     * screen position. If target is GL3.0 and up, this value is stored in the
     * out variable 'texcoord'.
     */
    GLuint
    passThroughVertexShader() const { return m_pass_through_vs; }

    /** Configure inputs for a program that links with the pass-through vertex shader. */
    bool
    configurePassThroughVertexShader( GLuint program ) const;

protected:
    HPMCConstants*  m_constants;
    GLuint          m_vbo;
    GLuint          m_vao;
    GLuint          m_pass_through_vs;
};

#endif // GPGPUQUAD_HPP
