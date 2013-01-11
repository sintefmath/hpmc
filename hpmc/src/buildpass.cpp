/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: buildpass.cpp
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

using std::cerr;
using std::endl;

// -----------------------------------------------------------------------------
bool
HPMCtriggerHistopyramidBuildPasses( struct HPMCHistoPyramid* h )
{
    if( h == NULL ) {
        return false;
    }
    HPMCHistoPyramid::HistoPyramid& hp = h->m_histopyramid;
    HPMCHistoPyramid::HistoPyramidBuild& hpb = h->m_hp_build;
    HPMCHistoPyramid::HistoPyramidBuild::BaseConstruction& base = hpb.m_base;
    HPMCHistoPyramid::HistoPyramidBuild::FirstReduction& first = hpb.m_first;
    HPMCHistoPyramid::HistoPyramidBuild::UpperReduction& upper = hpb.m_upper;

    // --- if we have errors already on state, we fail -------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: triggerHistopyramidBuildPasses called with GL errors." << endl;
#endif
        return false;
    }

    // --- build base level ----------------------------------------------------
    glUseProgram( base.m_program );

    // unless custom, HPMC handles fetching from the scalar field texture. We
    // bind the scalar field to the unit given by h->m_hp_build.m_tex_unit_2.
    if( h->m_fetch.m_mode == HPMC_VOLUME_LAYOUT_TEXTURE_3D ) {
        glActiveTextureARB( GL_TEXTURE0_ARB + hpb.m_tex_unit_2 );
        glBindTexture( GL_TEXTURE_3D, h->m_fetch.m_tex );
    }

    // Switch to texture unit given by h->m_hp_build.m_tex_unit_1.
    glActiveTextureARB( GL_TEXTURE0_ARB + hpb.m_tex_unit_1 );

    // To avoid getting GL errors when we bind base level FBOs, we set mipmap
    // levels of the HP texture to zero.
    glBindTexture( GL_TEXTURE_2D, h->m_histopyramid.m_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    // Then bind the vertex count texture to unit h->m_hp_build.m_tex_unit_1.
    glBindTexture( GL_TEXTURE_1D, h->m_constants->m_vertex_count_tex );

    // Update the threshold uniform
    glUniform1f( base.m_loc_threshold, h->m_threshold );

    // And trigger computation.
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, hp.m_fbos[0] );
    glViewport( 0, 0, hp.m_size, hp.m_size );
    HPMCrenderGPGPUQuad( h );

    // If HP is only 1x1 texels big, we are finished.
    if( hp.m_size_l2 < 1 ) {
        return true;
    }

    // --- first reduction of HP -----------------------------------------------
    glUseProgram( first.m_program );

    // bind histopyramid to current texture unit (h->m_hp_build.m_tex_unit_1),
    // max mipmap level is already set to zero.
    glBindTexture( GL_TEXTURE_2D, hp.m_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );

    // distance between texels in base layer of HP
    if( h->m_constants->m_target < HPMC_TARGET_GL30_GLSL130 ) {
        glUniform2f( first.m_loc_delta, -0.5f/hp.m_size, 0.5f/hp.m_size );
    }
    else {
        glUniform1i( first.m_loc_src_level, 0 );
    }

    // trigger
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid.m_fbos[1] );
    glViewport( 0, 0, hp.m_size/2, hp.m_size/2 );
    HPMCrenderGPGPUQuad( h );

    // If HP is only 2x2 texels big, we are finished.
    if( hp.m_size_l2 < 2 ) {
        return true;
    }

    // --- trigger the rest of reductions --------------------------------------
    glUseProgram( upper.m_program );

    if( h->m_constants->m_target < HPMC_TARGET_GL30_GLSL130 ) {
        for(GLsizei m=2; m<=hp.m_size_l2; m++) {

            // set legal mipmap levels to the previous level. The HP is still bound
            // to the current texture unit, so we don't need to bother with binding.
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, m-1 );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m-1 );

            // Distance between texels in the previous level.
            glUniform2f( upper.m_loc_delta, -0.5f/(1<<(hp.m_size_l2+1-m)), 0.5f/(1<<(hp.m_size_l2+1-m)) );

            // Trigger
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid.m_fbos[m] );
            glViewport( 0, 0, 1<<(hp.m_size_l2-m), 1<<(hp.m_size_l2-m) );
            HPMCrenderGPGPUQuad( h );
        }
    }
    else {
        for(GLsizei m=2; m<=h->m_histopyramid.m_size_l2; m++) {
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m-1 );
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid.m_fbos[m] );
            glViewport( 0, 0, 1<<(hp.m_size_l2-m), 1<<(hp.m_size_l2-m) );
            glUniform1i( upper.m_loc_src_level, m-1 );
            HPMCrenderGPGPUQuad( h );
        }

    }

    // --- trigger readback ----------------------------------------------------
    glBindBuffer( GL_PIXEL_PACK_BUFFER, hp.m_top_pbo );
    glBindTexture( GL_TEXTURE_2D, hp.m_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, hp.m_size_l2 );
    glGetTexImage( GL_TEXTURE_2D, hp.m_size_l2, GL_RGBA, GL_FLOAT, NULL );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
    hp.m_top_count_updated = false;

    // --- if we have created errors, we fail ----------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: triggerHistopyramidBuildPasses produced GL errors." << endl;
#endif
        return false;
    }
    return true;
}
