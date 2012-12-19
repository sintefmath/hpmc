#ifndef ISOSURFACE_HPP
#define ISOSURFACE_HPP
#include <hpmc_internal.h>
#include "BaseLevelBuilder.hpp"
#include "HistoPyramid.hpp"
#include "Field.hpp"

// -----------------------------------------------------------------------------
/** A HistoPyramid for a particular volume configuration. */
struct HPMCIsoSurface
{
public:
    HPMCIsoSurface(HPMCConstants *constants );

    ~HPMCIsoSurface();

    bool
    init();

    void
    taint();

    bool
    isBroken() const { return m_broken; }

    void
    setAsBroken();

    bool
    untaint();

    const Field&
    field() const { return m_field; }

    Field&
    field() { return m_field; }

    const HPMCConstants*
    constants() const { return m_constants; }

    const HPMCBaseLevelBuilder&
    baseLevelBuilder() const { return m_base_builder; }

    const HPMCHistoPyramid&
    histoPyramid() const { return m_histopyramid; }

    GLfloat
    threshold() const { return m_threshold; }

    void
    setThreshold( GLfloat threshold ) { m_threshold = threshold; }

    void
    build();

    GLsizei
    vertexCount();


    /** State during HistoPyramid construction */
    struct HistoPyramidBuild {
        GLuint           m_tex_unit_1;          ///< Bound to vertex count in base level pass, bound to HP in other passes.
        GLuint           m_tex_unit_2;          ///< Bound to volume texture if HPMC handles texturing of scalar field.
    }
    m_hp_build;
protected:
    bool                    m_tainted;   ///< HP needs to be rebuilt.
    bool                    m_broken;    ///< True if misconfigured, fails until reconfiguration.
    struct HPMCConstants*   m_constants;
    GLfloat                 m_threshold; ///< Cache to hold the threshold value used to build the HP.
    Field                   m_field;
    HPMCBaseLevelBuilder    m_base_builder;
    HPMCHistoPyramid        m_histopyramid;

};

#endif // ISOSURFACE_HPP
