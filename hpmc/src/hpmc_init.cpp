/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc_init.cpp
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

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <hpmc.h>
#include <hpmc_internal.h>

using std::min;
using std::max;

#ifdef _WIN32
  #define log2f(x) (logf(x)*1.4426950408889634f)
#endif

/** Determines the size and layout of the histopyramid based on volume size.
  *
  * The HistoPyramid is 2D while the volume is 3D. This function determines the
  * mapping between 2D and 3D and determines the resulting size of the
  * HistoPyramid.
  *
  * Currently, the layout algorithm is quite naive, it is suboptimal for
  * non-square tile sizes. The basic requirement is that the full size of the
  * base level is a power of two, and otherwise as small as possible.
  *
  * It is assumed that m_volume_width, m_volume_height, and m_volume_depth has
  * been initialized and m_base_tile_width, m_base_tile_height, m_base_cols,
  * m_base_rows, m_base_size_l2, and m_base_size is set.
  *
  * Does not call the OpenGL API.
  */
void
HPMCsetLayout( struct HPMCHistoPyramid* h )
{
    // currently, the layout algorithm is quite naive, it is
    // suboptimal for non-square tile sizes. The basic requirement is
    // that the full size of the base level is a power of two, and
    // otherwise as small as possible.

    h->m_base_tile_width = 1u<<(GLsizei)ceilf( log2f( (float)(h->m_volume_width-1u)/2.0 ) );
    h->m_base_tile_height = 1u<<(GLsizei)ceilf( log2f( (float)(h->m_volume_height-1u)/2.0 ) );

    float aspect = (float)h->m_base_tile_width/(float)h->m_base_tile_height;
    std::cerr << "HPMC: tile aspect ratio = " << aspect << "\n";

    // determine mapping between HP 2D layout and volume 3D layout
    h->m_base_cols = 1u<<
            (GLsizei)max( 0.0f,
                          ceilf( log2f( sqrtf( (float)(h->m_volume_depth-1u)/aspect ) ) ) );
    h->m_base_rows = (h->m_volume_depth+h->m_base_cols-2u)/h->m_base_cols;

    // determine size of base level (and height of pyramid)
    h->m_base_size_l2 =
        (GLsizei)ceilf( log2f( (float)max( h->m_base_tile_width  * h->m_base_cols,
                                           h->m_base_tile_height * h->m_base_rows ) ) );
    h->m_base_size = 1<<h->m_base_size_l2;

    // determine actual set of rows and columns
    h->m_base_cols = h->m_base_size / h->m_base_tile_width;
    h->m_base_rows = h->m_base_size / h->m_base_tile_height;


#ifdef DEBUG
    std::cerr << "HPMC: volume_width = " << h->m_volume_width << std::endl;
    std::cerr << "HPMC: volume_height = " << h->m_volume_height << std::endl;
    std::cerr << "HPMC: volume_depth = " << h->m_volume_depth << std::endl;
    std::cerr << "HPMC: base_tile_width = " << h->m_base_tile_width << std::endl;
    std::cerr << "HPMC: base_tile_height = " << h->m_base_tile_height << std::endl;
    std::cerr << "HPMC: base_tile_cols = " << h->m_base_cols << std::endl;
    std::cerr << "HPMC: base_tile_rows = " << h->m_base_rows << std::endl;
    std::cerr << "HPMC: base_size_l2 = " << h->m_base_size_l2 << std::endl;
    std::cerr << "HPMC: base_sizes = " << h->m_base_size << std::endl;
#endif

}

/** Creates the HistoPyramid texture and framebuffer object.
  *
  * \sideeffect Taints the current texture and current frame buffer object.
  */
void
HPMCcreateHistopyramid( struct HPMCHistoPyramid* h )
{
    // create histopyramid texture
    glGenTextures( 1, &h->m_histopyramid_tex );
    glBindTexture( GL_TEXTURE_2D, h->m_histopyramid_tex );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB,
                  h->m_base_size, h->m_base_size, 0,
                  GL_RGBA, GL_FLOAT, NULL );
    glGenerateMipmapEXT( GL_TEXTURE_2D );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, h->m_base_size_l2 );
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }

    // create histopyramid framebuffer objects
    h->m_histopyramid_fbos = new GLuint[ h->m_base_size_l2+1 ];
    glGenFramebuffersEXT( h->m_base_size_l2+1, h->m_histopyramid_fbos );
    for( GLuint m=0; m<=h->m_base_size_l2; m++) {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid_fbos[m] );
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
                                   GL_COLOR_ATTACHMENT0_EXT,
                                   GL_TEXTURE_2D,
                                   h->m_histopyramid_tex,
                                   m );
        glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
        if( !HPMCcheckFramebufferStatus( __FILE__, __LINE__ ) ) {
            abort();
        }
    }

}

void
HPMCcreateShaderPrograms( struct HPMCHistoPyramid* h )
{
    // create shaders
    h->m_gpgpu_passthrough_v = HPMCcompileShader( HPMCgenerateDefines(h) +
                                                  HPMCgenerateGPGPUVertexPassThroughShader(h),
                                                  GL_VERTEX_SHADER );
    h->m_baselevel_f = HPMCcompileShader( HPMCgenerateDefines(h) +
                                          HPMCgenerateScalarFieldFetch(h) +
                                          HPMCgenerateBaselevelShader(h),
                                          GL_FRAGMENT_SHADER );
    h->m_baselevel_p = glCreateProgram();
    glAttachShader( h->m_baselevel_p, h->m_gpgpu_passthrough_v );
    glAttachShader( h->m_baselevel_p, h->m_baselevel_f );
    HPMClinkProgram( h->m_baselevel_p );
    h->m_baselevel_threshold_loc = HPMCgetUniformLocation( h->m_baselevel_p, "threshold" );

    glUseProgram( h->m_baselevel_p );
    glUniform1i( HPMCgetUniformLocation( h->m_baselevel_p, "HPMC_scalarfield" ), 0 );
    glUniform1i( HPMCgetUniformLocation( h->m_baselevel_p, "vertexcount" ), 1 );

    h->m_first_reduction_f = HPMCcompileShader( HPMCgenerateDefines(h) +
                                                HPMCgenerateReductionShader( h, "floor" ),
                                                GL_FRAGMENT_SHADER );
    h->m_first_reduction_p = glCreateProgram();
    glAttachShader( h->m_first_reduction_p, h->m_gpgpu_passthrough_v );
    glAttachShader( h->m_first_reduction_p, h->m_first_reduction_f );
    HPMClinkProgram( h->m_first_reduction_p );
    h->m_first_reduction_delta_loc =
            HPMCgetUniformLocation( h->m_first_reduction_p, "delta" );

    glUseProgram( h->m_first_reduction_p );
    glUniform1i( HPMCgetUniformLocation( h->m_first_reduction_p, "histopyramid" ), 0 );

    h->m_reduction_f = HPMCcompileShader( HPMCgenerateDefines(h) +
                                          HPMCgenerateReductionShader( h ),
                                          GL_FRAGMENT_SHADER );
    h->m_reduction_p = glCreateProgram();
    glAttachShader( h->m_reduction_p, h->m_gpgpu_passthrough_v );
    glAttachShader( h->m_reduction_p, h->m_reduction_f );
    HPMClinkProgram( h->m_reduction_p );
    h->m_reduction_delta_loc =
            HPMCgetUniformLocation( h->m_reduction_p, "delta" );

    glUseProgram( h->m_reduction_p );
    glUniform1i( HPMCgetUniformLocation( h->m_reduction_p, "histopyramid" ), 0 );

}
