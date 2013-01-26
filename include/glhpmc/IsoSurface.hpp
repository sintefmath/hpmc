#pragma once
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
#include <glhpmc/glhpmc_internal.hpp>
#include <glhpmc/BaseLevelBuilder.hpp>
#include <glhpmc/HistoPyramid.hpp>
#include <glhpmc/Field.hpp>

namespace glhpmc {

// -----------------------------------------------------------------------------
/** A HistoPyramid for a particular volume configuration. */
struct HPMCIsoSurface
{
public:

    /** Creates a new HistoPyramid instance on the current context.
      *
      * \param s  A pointer to a constant instance residing on a context sharing
      *           resources with the current context.
      *
      * \return   A new HistoPyramid instance.
      *
      * \sideeffect None.
      */
    static
    HPMCIsoSurface*
    factory( HPMCConstants*     constants,
             Field*             field,
             bool               binary = false,
             unsigned int       cells_x = 0,
             unsigned int       cells_y = 0,
             unsigned int       cells_z = 0 );

    ~HPMCIsoSurface();


    void
    build( GLfloat iso );

    GLsizei
    vertexCount();

    GLuint
    builderProgram();


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

    const Field*
    field() const { return m_field; }

    const HPMCConstants*
    constants() const { return m_constants; }

    const HPMCBaseLevelBuilder&
    baseLevelBuilder() const { return m_base_builder; }

    const HPMCHistoPyramid&
    histoPyramid() const { return m_histopyramid; }

    GLfloat
    threshold() const { return m_threshold; }

    unsigned int
    cellsX() const { return m_cells_x; }

    unsigned int
    cellsY() const { return m_cells_x; }

    unsigned int
    cellsZ() const { return m_cells_x; }

    bool
    binary() const { return m_binary; }

    /** State during HistoPyramid construction */
    struct HistoPyramidBuild {
        GLuint           m_tex_unit_1;          ///< Bound to vertex count in base level pass, bound to HP in other passes.
        GLuint           m_tex_unit_2;          ///< Bound to volume texture if HPMC handles texturing of scalar field.
    }
    m_hp_build;

private:
    bool            m_tainted;   ///< HP needs to be rebuilt.
    bool            m_broken;    ///< True if misconfigured, fails until reconfiguration.
    HPMCConstants*  m_constants;
    Field*          m_field;
    unsigned int    m_cells_x;
    unsigned int    m_cells_y;
    unsigned int    m_cells_z;
    bool            m_binary;

    GLfloat         m_threshold; ///< Cache to hold the threshold value used to build the HP.

    HPMCBaseLevelBuilder    m_base_builder;
    HPMCHistoPyramid        m_histopyramid;

    HPMCIsoSurface( HPMCConstants*  constants,
                    Field*          field,
                    bool            binary,
                    unsigned int    cells_x,
                    unsigned int    cells_y,
                    unsigned int    cells_z );

};

} // of namespace glhpmc
