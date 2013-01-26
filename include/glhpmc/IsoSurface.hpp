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
      * \return   A new HistoPyramid instance.
      *
      * \sideeffect None.
      */
    static
    HPMCIsoSurface*
    factory( HPMCConstants* s );

    ~HPMCIsoSurface();

    /** Specify size of scalar field lattice.
      *
      * Specify the number of scalar field samples along the x,y,z-directions. If
      * using a 3D texture, this is the same as the size of the texture.
      *
      * \param h       Pointer to an existing HistoPyramid instance.
      * \param x_size  The size of the lattice along the x-axis.
      * \param y_size  The size of the lattice along the y-axis.
      * \param z_size  The size of the lattice along the z-axis.
      *
      * \sideeffect Triggers rebuilding of shaders and textures.
      */
    void
    setLatticeSize( GLsizei x_size, GLsizei y_size, GLsizei z_size );

    /** Specify the number of cells in the grid of Marching Cubes cells.
      *
      * Since the cells reside in-between the scalar field lattice points, the
      * default size is lattice size - 1. If the gradient is not given, it is
      * approximated using forward differences. In this case, the scalar field
      * is sampled outside the lattice, giving shading artefacts along three of the
      * faces of the domain. Reducing grid size to lattice size - 2 removes this
      * artefact.
      *
      * \param h       Pointer to an existing HistoPyramid instance.
      * \param x_size  The size of the grid along the x-axis.
      * \param y_size  The size of the grid along the y-axis.
      * \param z_size  The size of the grid along the z-axis.
      *
      * \sideeffect Triggers rebuilding of shaders and textures.
      */
    void
    setGridSize( GLsizei x_size, GLsizei y_size, GLsizei z_size );

    /** Specify the extent of the grid in object space.
      *
      * This specifies the grid size in object space, defaults to (1.0,1.0,1.0).
      *
      * \param h       Pointer to an existing HistoPyramid instance.
      * \param x_size  The size of the grid along the x-axis.
      * \param y_size  The size of the grid along the y-axis.
      * \param z_size  The size of the grid along the z-axis.
      *
      * \sideeffect Triggers rebuilding of shaders and textures.
      */
    void
    setGridExtent(GLsizei x_extent, GLsizei y_extent, GLsizei z_extent );


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
    build( GLfloat iso );


    GLsizei
    vertexCount();


    /** State during HistoPyramid construction */
    struct HistoPyramidBuild {
        GLuint           m_tex_unit_1;          ///< Bound to vertex count in base level pass, bound to HP in other passes.
        GLuint           m_tex_unit_2;          ///< Bound to volume texture if HPMC handles texturing of scalar field.
    }
    m_hp_build;

private:
    bool                    m_tainted;   ///< HP needs to be rebuilt.
    bool                    m_broken;    ///< True if misconfigured, fails until reconfiguration.
    struct HPMCConstants*   m_constants;
    GLfloat                 m_threshold; ///< Cache to hold the threshold value used to build the HP.
    Field                   m_field;
    HPMCBaseLevelBuilder    m_base_builder;
    HPMCHistoPyramid        m_histopyramid;

    HPMCIsoSurface(HPMCConstants *constants );



};

} // of namespace glhpmc
