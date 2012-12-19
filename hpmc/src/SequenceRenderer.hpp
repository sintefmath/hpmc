#ifndef SEQUENCERENDERER_HPP
#define SEQUENCERENDERER_HPP
#include <hpmc.h>

/** Draw a sequence of enumerated vertices.
 *
 * When extracting the iso-surface, we need to know the vertex number (which is
 * used as key when traversing the histopyramid). On GL 3.0 and up, gl_VertexID
 * provides that information. On lower versions, we fake gl_Vertex by drawing
 * from a vbo where the x-component is an increasing sequence.
 */
class HPMCSequenceRenderer
{
public:
    /** Constructor.
     *
     * \sideeffect GL_ARRAY_BUFFER (glsl < 130).
     */
    HPMCSequenceRenderer( HPMCConstants* constants );

    ~HPMCSequenceRenderer( );

    void
    init();

    /** Sets up vertex input for enumerated vertex drawing.
     *
     * On GL 3.0 and up, this function does nothing. On lower versions, it binds
     * a vertex array.
     */
    void
    bindVertexInputs() const;

    /** Draw a given number of enumerated vertices.
     *
     * \param[in] offset_loc  Uniform location for currently bound shader where
     *                        an offset uniform integer is stored (used for
     *                        splitting a draw into several batches). Not used
     *                        for GL 3.0 and up.
     * \param[in] num         The number of primitives.
     */
    void
    render( GLint offset_loc, GLsizei num ) const;

    void
    unbindVertexInputs() const;

protected:
    HPMCConstants*  m_constants;
    GLsizei         m_batch_size;
    GLuint          m_vbo;
    GLuint          m_vao;  // Empty VAO

};


#endif // SEQUENCERENDERER_HPP
