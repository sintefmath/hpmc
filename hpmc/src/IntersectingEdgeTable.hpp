#ifndef INTERSECTINGEDGETABLE_HPP
#define INTERSECTINGEDGETABLE_HPP
#include <hpmc.h>

/** Marching Cubes table with vertices encoded as edge intersections.
 *
 * Each vertex produced by the marching cubes is located on a cell edge. This
 * table encodes which cell edge that vertex is on.
 */
class HPMCIntersectingEdgeTable
{
public:
    HPMCIntersectingEdgeTable( HPMCConstants* constants );

    ~HPMCIntersectingEdgeTable();

    /** Initializes object.
     *
     * \sideeffect active texture,
     *             texture binding
     */
    void
    init( );

    /** Normal table. */
    GLuint
    texture() const { return m_tex; }

    /** Same as \ref texture, but with orientation of triangles flipped. */
    GLuint
    textureFlip() const { return m_tex_flip; }

    /** Same as \ref texture, but with normal vectors for binary fields encoded in fractional parts of x,y,z. */
    GLuint
    textureNormal() const { return m_tex_normal; }

    /** Same as \ref textureNormal, but with orientation of triangles flipped. */
    GLuint
    textureNormalFlip() const { return m_tex_normal_flip; }

protected:
    HPMCConstants*  m_constants;
    GLuint          m_tex;
    GLuint          m_tex_flip;
    GLuint          m_tex_normal;
    GLuint          m_tex_normal_flip;
};
#endif // INTERSECTINGEDGETABLE_HPP
