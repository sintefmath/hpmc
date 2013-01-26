/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
 *
 * Author(s): Christopher Dyken, <christopher.dyken@sintef.no>
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
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP
#include <glhpmc/glhpmc_internal.hpp>
#include <glhpmc/SequenceRenderer.hpp>
#include <glhpmc/VertexCountTable.hpp>
#include <glhpmc/GPGPUQuad.hpp>
#include <glhpmc/IntersectingEdgeTable.hpp>

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
