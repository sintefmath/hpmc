/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc_internal.h
 *
 *  Created: 24. June 2009
 *
 *  Version: $Id: $
 *
 *  Authors: Christopher Dyken <christopher.dyken@sintef.no>
 *
 *  This file is part of the HPMC library.
 *  Copyright (C) 2009 by SINTEF.  All rights reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  ("GPL") version 2 as published by the Free Software Foundation.
 *  See the file LICENSE.GPL at the root directory of this source
 *  distribution for additional information about the GNU GPL.
 *
 *  For using HPMC with software that can not be combined with the
 *  GNU GPL, please contact SINTEF for aquiring a commercial license
 *  and support.
 *
 *  SINTEF, Pb 124 Blindern, N-0314 Oslo, Norway
 *  http://www.sintef.no
 *********************************************************************/
/**
  * \file hpmc_internal.h
  *
  * \brief Library internal header file. Defines internal interface.
  *
  *
  */
#ifndef _HPMC_INTERNAL_H_
#define _HPMC_INTERNAL_H_

#include <GL/glew.h>
#include <string>
#include <vector>

/** \addtogroup hpmc_public
  * \{
  */

// -----------------------------------------------------------------------------
enum HPMCVolumeLayout {
    HPMC_VOLUME_LAYOUT_CUSTOM,
    HPMC_VOLUME_LAYOUT_TEXTURE_3D
};

// -----------------------------------------------------------------------------
/** Constant data shared by multiple HistoPyramids. */
struct HPMCConstants
{
    GLuint            m_vertex_count_tex;
    GLuint            m_edge_decode_tex;
    GLuint            m_enumerate_vbo;
    GLsizei           m_enumerate_vbo_n;
    GLuint            m_gpgpu_quad_vbo;
};

// -----------------------------------------------------------------------------
/** A HistoPyramid for a particular volume configuration. */
struct HPMCHistoPyramid
{
    /** Tag that the HP is changes such that shaders etc must be rebuilt. */
    bool                   m_tainted;
    /** Tag that we have had an error and all entry points should return until
      * HP is reconfigured.
      */
    bool                   m_broken;
    /** Pointer to a set of constants on this context. */
    struct HPMCConstants*  m_constants;
    /** Cache to hold the threshold value used to build the HP. */
    GLfloat                m_threshold;

    // -------------------------------------------------------------------------
    /** Specifies how the base level of the HistoPyramid is laid out. */
    struct Tiling {
        /** The size of a tile in the base level. */
        GLsizei       m_tile_size[2];
        /** The number of tiles in the base level along the x and y direction. */
        GLsizei       m_layout[2];
    }
    m_tiling;

    // -------------------------------------------------------------------------
    /** Information about the HistoPyramid texture. */
    struct HistoPyramid {
        /** The size of the HP tex.
          *
          * The tex is quadratic, so the size is the same along x and y.
          */
        GLsizei              m_size;
        /** The two-log of the size of the HP tex.
          *
          * The tex size is always a power-of-two, so this is always an integer.
          */
        GLsizei              m_size_l2;
        /** Texture name of the HP tex. */
        GLuint               m_tex;
        /** A set of FBOs, one FBO per mipmap level in the HP tex. */
        std::vector<GLuint>  m_fbos;
        /** Pixel pack buffer for async readback of HP top element. */
        GLuint               m_top_pbo;
        /** Cache result of readback of the HP top element PBO. */
        GLsizei              m_top_count;
        /** Tag that the cached result is valid, so PBO need not to be consulted. */
        GLsizei              m_top_count_updated;
    }
    m_histopyramid;

    // -------------------------------------------------------------------------
    /** Specifies the layout of the scalar field. */
    struct Field {
        /** The x,y,z-size of the lattice of scalar field samples. */
        GLsizei       m_size[3];
        /** The x,y,z-size of the MC grid, defaults to m_size-[1,1,1]. */
        GLsizei       m_cells[3];
        /** The extent of the MC grid when outputted from the traversal shader. */
        GLfloat       m_extent[3];
    }
    m_field;

    // -------------------------------------------------------------------------
    /** Specifies how data is fetched from the scalar field. */
    struct Fetch {
        /** Specifies what type of fetching is used.
          *
          * Currently supported is fetching from a Texture3D or using a custom
          * shader function.
          */
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
    }
    m_fetch;

    /** State during HistoPyramid construction */
    struct HistoPyramidBuild {
        GLuint           m_tex_unit_1;          ///< Bound to vertex count in base level pass, bound to HP in other passes.
        GLuint           m_tex_unit_2;          ///< Bound to volume texture if HPMC handles texturing of scalar field.
        GLuint           m_gpgpu_vertex_shader; ///< Common GPGPU pass-through vertex shader.

        /** Base level construction pass. */
        struct BaseConstruction {
            GLuint            m_fragment_shader;
            GLuint            m_program;
            GLint             m_loc_threshold;
        }
        m_base;

        /** First pure reduction pass. */
        struct FirstReduction {
            GLuint            m_fragment_shader;
            GLuint            m_program;
            GLint             m_loc_delta;
        }
        m_first;

       /** First pure reduction pass. */
        struct UpperReduction {
            GLuint            m_fragment_shader;
            GLuint            m_program;
            GLint             m_loc_delta;
        }
        m_upper;

    }
    m_hp_build;
};

// -----------------------------------------------------------------------------
struct HPMCTraversalHandle
{
    struct HPMCHistoPyramid*  m_handle;
    GLuint                    m_program;
    GLuint                    m_scalarfield_unit;
    GLuint                    m_histopyramid_unit;
    GLuint                    m_edge_decode_unit;
    GLint                     m_offset_loc;
    GLint                     m_threshold_loc;
};

/** \} */
// -----------------------------------------------------------------------------
/** \defgroup hpmc_internal Internal API
  * \{
  */


extern int      HPMC_triangle_table[256][16];

extern GLfloat  HPMC_edge_table[12][4];

extern GLfloat  HPMC_gpgpu_quad_vertices[3*4];

/** Sets up hp textures and shaders.
  *
  * \sideeffect GL_CURRENT_PROGRAM,
  *             GL_TEXTURE_2D_BINDING,
  *             GL_FRAMEBUFFER_BINDING
  */
bool
HPMCsetup( struct HPMCHistoPyramid* h );

/** Checks field and grid sizes and determine HistoPyramid layout and tiling.
  *
  * \sideeffect None.
 */
bool
HPMCdetermineLayout( struct HPMCHistoPyramid* h );

/** Creates the HistoPyramid texture and framebuffer object.
  *
  * \sideeffect GL_TEXTURE_2D_BINDING, GL_FRAMEBUFFER_BINDING
  */
bool
HPMCsetupTexAndFBOs( struct HPMCHistoPyramid* h );

bool
HPMCfreeHPBuildShaders( struct HPMCHistoPyramid* h );

/** Build reduction shaders.
  *
  * \sideeffect GL_CURRENT_PROGRAM
  */
bool
HPMCbuildHPBuildShaders( struct HPMCHistoPyramid* h );


bool
HPMCcheckFramebufferStatus( const std::string& file, const int line );

bool
HPMCcheckGL( const std::string& file, const int line );

std::string
HPMCaddLineNumbers( const std::string& src );

GLuint
HPMCcompileShader( const std::string& src, GLuint type );

bool
HPMClinkProgram( GLuint program );

GLint
HPMCgetUniformLocation( GLuint program, const std::string& name );

std::string
HPMCgenerateDefines( struct HPMCHistoPyramid* h );

std::string
HPMCgenerateScalarFieldFetch( struct HPMCHistoPyramid* h );

std::string
HPMCgenerateBaselevelShader( struct HPMCHistoPyramid* h );

std::string
HPMCgenerateReductionShader( struct HPMCHistoPyramid* h, const std::string& filter="" );

std::string
HPMCgenerateGPGPUVertexPassThroughShader( struct HPMCHistoPyramid* h );

std::string
HPMCgenerateExtractVertexFunction( struct HPMCHistoPyramid* h );


/** Trigger computations that build the Histopyramid.
  *
  * Evaluates the scalar field, determines codes and vertex counts and builds the HP base layer.
  *
  * \sideeffect Active texture unit,
  *             two texture units (see h->m_base_level,m_tex_units..),
  *             GL_CURRENT_PROGRAM,
  *             GL_FRAMEBUFFER_BINDING,
  *             GL_VIEWPORT,
  *             GL_VERTEX_ARRAY,
  *             GL_VERTEX_ARRAY_SIZE,
  *             GL_VERTEX_ARRAY_TYPE,
  *             GL_VERTEX_ARRAY_STRIDE,
  *             GL_VERTEX_ARRAY_POINTER.
  *             GL_PIXEL_PACK_BUFFER binding
  */
bool
HPMCtriggerHistopyramidBuildPasses( struct HPMCHistoPyramid* h );


void
HPMCsetLayout( struct HPMCHistoPyramid* h );

/** Renders a GPGPU quad from a VBO.
  *
  * \sideeffect GL_VERTEX_ARRAY,
  *             GL_VERTEX_ARRAY_SIZE,
  *             GL_VERTEX_ARRAY_TYPE,
  *             GL_VERTEX_ARRAY_STRIDE,
  *             GL_VERTEX_ARRAY_POINTER.
  */
void
HPMCrenderGPGPUQuad( struct HPMCHistoPyramid* h );

/** \} */

#endif // _HPMC_INTERNAL_H_
