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

    const std::string
    downTraversalSource() const;

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

    /** Create fragment shader source for reductions
     *
     * \param first  True if shader is first reduction, false for subsequent
     *               reductions.
     */
    const std::string
    fragmentSource( bool first ) const;


};

#endif // HISTOPYRAMID_HPP
