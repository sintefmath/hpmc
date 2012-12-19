#ifndef BASELEVELBUILDER_HPP
#define BASELEVELBUILDER_HPP
#include <hpmc.h>
#include "Field.hpp"

class HPMCIsoSurface;

/** Specifies how the base level of the HistoPyramid is laid out. */
class HPMCBaseLevelBuilder
{
public:
    HPMCBaseLevelBuilder( const HPMCIsoSurface* iso_surface );

    ~HPMCBaseLevelBuilder();

    bool
    configure();

    bool
    build( GLuint vertex_table_sampler, GLuint field_sampler );


    /** Returns the shader program used to build the base level. */
    GLuint
    program() const { return m_program; }

    GLsizei
    log2Size() const { return m_size_l2; }

    GLsizei
    layoutX() const { return m_layout[0]; }

    GLsizei
    layoutY() const { return m_layout[1]; }

    GLsizei
    tileSizeX() const { return m_tile_size[0]; }

    GLsizei
    tileSizeY() const { return m_tile_size[1]; }

    GLuint              m_program;
    GLint               m_loc_threshold;
    Field::Context      m_field_context;

    const std::string
    extractSource() const;

protected:
    const HPMCIsoSurface*   m_iso_surface;
    GLsizei                 m_tile_size[2]; ///< The size of a tile in the base level.
    GLsizei                 m_layout[2];    ///< Number of tiles along x and y in the base level.
    GLsizei                 m_size_l2;      ///< The full log2-size of the base level.

    GLint                   m_loc_vertex_count_table;

    const std::string
    fragmentSource() const;

};


#endif // BASELEVELBUILDER_HPP
