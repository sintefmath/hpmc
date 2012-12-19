#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP
#include <hpmc_internal.h>
#include "SequenceRenderer.hpp"
#include "VertexCountTable.hpp"
#include "GPGPUQuad.hpp"
#include "IntersectingEdgeTable.hpp"

struct HPMCConstants
{
public:
    HPMCConstants( HPMCTarget target, HPMCDebugBehaviour debug );

    ~HPMCConstants();

    HPMCDebugBehaviour
    debugBehaviour() const { return m_debug; }

    void
    init();

    /** Return the GLSL version declaration for the current target. */
    const std::string&
    versionString() const { return m_version_string; }

    const HPMCTarget
    target() const { return m_target; }

    const HPMCVertexCountTable&
    caseVertexCounts() const { return m_vertex_counts; }

    const HPMCGPGPUQuad&
    gpgpuQuad() const { return m_gpgpu_quad; }

    const HPMCSequenceRenderer&
    sequenceRenderer() const { return m_sequence_renderer; }

    const HPMCIntersectingEdgeTable&
    edgeTable() const { return m_edge_table; }

protected:
    HPMCTarget                  m_target;
    HPMCDebugBehaviour          m_debug;
    HPMCVertexCountTable        m_vertex_counts;
    HPMCGPGPUQuad               m_gpgpu_quad;
    HPMCSequenceRenderer        m_sequence_renderer;
    HPMCIntersectingEdgeTable   m_edge_table;
    std::string                 m_version_string;
};

#endif // CONSTANTS_HPP
