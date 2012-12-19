#ifndef FIELD_HPP
#define FIELD_HPP
#include <string>
#include <hpmc.h>

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
    hasGradient() const { return m_gradient; }

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
    /** True if the texture or the shader function can provide gradients.
      *
      * If gradients are provided, they are used to find normal vectors,
      * it not, forward differences are used.
      */
    bool              m_gradient;

protected:
    HPMCConstants*  m_constants;

};

#endif // FIELD_HPP
