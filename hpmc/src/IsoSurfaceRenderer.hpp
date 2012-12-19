#ifndef ISOSURFACERENDERER_HPP
#define ISOSURFACERENDERER_HPP
#include <hpmc.h>
#include <string>

struct HPMCIsoSurfaceRenderer
{
public:
    HPMCIsoSurfaceRenderer( struct HPMCIsoSurface* iso_surface );

    ~HPMCIsoSurfaceRenderer();

    const std::string
    extractionSource() const;

    bool
    setProgram( GLuint program,
                GLuint tex_unit_work1,
                GLuint tex_unit_work2,
                GLuint tex_unit_work3 );

    bool
    draw( int transform_feedback_mode, bool flip_orientation );

    struct HPMCIsoSurface*  m_handle;
    GLuint                    m_program;
    GLuint                    m_scalarfield_unit;
    GLuint                    m_histopyramid_unit;
    GLuint                    m_edge_decode_unit;
    GLint                     m_offset_loc;
    GLint                     m_threshold_loc;

protected:

};


#endif // ISOSURFACERENDERER_HPP
