#ifndef VERTEXCOUNTTABLE_HPP
#define VERTEXCOUNTTABLE_HPP
#include <hpmc.h>


/** Table containing the number of vertices needed for each Marching Cubes case. */
class HPMCVertexCountTable
{
public:
    HPMCVertexCountTable( HPMCConstants* constants );

    ~HPMCVertexCountTable( );

    /** Initializes object.
     *
     * \sideeffect active texture,
     *             texture binding
     */
    void
    init( );

    GLuint
    texture() const { return m_texture; }

protected:
    HPMCConstants*  m_constants;
    GLuint          m_texture;
};


#endif // VERTEXCOUNTTABLE_HPP
