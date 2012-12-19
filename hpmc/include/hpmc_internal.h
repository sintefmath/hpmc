/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc.h
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

#include <string>

/** \addtogroup hpmc_public
  * \{
  */

/** Constant data shared by multiple HistoPyramids.
  *
  */
struct HPMCConstants
{
    GLuint            m_vertex_count_tex;
    GLuint            m_edge_decode_tex;
    GLuint            m_enumerate_vbo;
    GLsizei           m_enumerate_vbo_n;

    GLuint            m_gpgpu_quad_vbo;
};

/** A HistoPyramid for a particular volume configuration.
  *
  */
struct HPMCHistoPyramid
{
    struct HPMCConstants* m_constants;
    GLsizei               m_volume_width;
    GLsizei               m_volume_height;
    GLsizei               m_volume_depth;
    GLint                 m_volume_type;
    HPMCVolumeLayout      m_volume_layout;
    bool                  m_func_discrete_gradient;
    bool                  m_func_omit_boundary;

    GLuint            m_volume_tex;
    GLfloat           m_threshold;

    GLsizei           m_base_tile_width;
    GLsizei           m_base_tile_height;

    GLsizei           m_base_cols;
    GLsizei           m_base_rows;
    GLsizei           m_base_size_l2;
    GLsizei           m_base_size;
    GLuint            m_histopyramid_tex;
    GLuint*           m_histopyramid_fbos;

    GLuint            m_gpgpu_passthrough_v;
    GLuint            m_baselevel_f;
    GLuint            m_baselevel_p;
    GLint             m_baselevel_threshold_loc;

    GLuint            m_first_reduction_f;
    GLuint            m_first_reduction_p;
    GLint             m_first_reduction_delta_loc;

    GLuint            m_reduction_f;
    GLuint            m_reduction_p;
    GLint             m_reduction_delta_loc;

    GLuint            m_state_shader;
    GLuint            m_state_fbo;

    std::string       m_fetch_shader;
    GLuint            m_build_first_tex_unit;

};

struct HPMCTraversalHandle
{
    struct HPMCHistoPyramid*  m_handle;
    GLuint              m_program;
    GLuint              m_scalarfield_unit;
    GLuint              m_histopyramid_unit;
    GLuint              m_edge_decode_unit;
    GLint               m_offset_loc;
    GLint               m_threshold_loc;
};

/** \} */
/** \defgroup hpmc_internal Internal API
  * \{
  */

void
HPMCpushState( struct HPMCHistoPyramid* h );

void
HPMCpopState( struct HPMCHistoPyramid* h );

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

GLuint
HPMCbuildVertexCountTable( );

GLuint
HPMCbuildEdgeDecodeTable( );

GLuint
HPMCbuildGPGPUQuadVBO( );

GLuint
HPMCbuildEnumerateVBO( GLsizei vertices );

void
HPMCbuildHistopyramidBaselevel( struct HPMCHistoPyramid* h );

void
HPMCbuildHistopyramidFirstLevel( struct HPMCHistoPyramid* h );

void
HPMCbuildHistopyramidUpperLevels( struct HPMCHistoPyramid* h );

void
HPMCsetLayout( struct HPMCHistoPyramid* h );

void
HPMCcreateHistopyramid( struct HPMCHistoPyramid* h );

void
HPMCcreateShaderPrograms( struct HPMCHistoPyramid* h );

void
HPMCrenderGPGPUQuad( struct HPMCHistoPyramid* h );

/** \} */

#endif // _HPMC_INTERNAL_H_
