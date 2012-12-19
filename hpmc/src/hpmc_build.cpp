/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc_build.cpp
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

void
HPMCbuildHistopyramidBaselevel( struct HPMCHistoPyramid* h )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        std::cerr << "HPMC: warning: entered " << __FUNCTION__ << " with GL errors.\n";
        abort();
    }
    glUseProgram( h->m_baselevel_p );

    glActiveTextureARB( GL_TEXTURE0_ARB );
    glBindTexture( GL_TEXTURE_2D, h->m_histopyramid_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    glBindTexture( GL_TEXTURE_3D, h->m_volume_tex );
    glActiveTextureARB( GL_TEXTURE1_ARB );
    glBindTexture( GL_TEXTURE_1D, h->m_constants->m_vertex_count_tex );
    glUniform1f( h->m_baselevel_threshold_loc, h->m_threshold );

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid_fbos[0] );
    glViewport( 0, 0, 1<<h->m_base_size_l2, 1<<h->m_base_size_l2 );

    HPMCrenderGPGPUQuad( h );

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        std::cerr << "HPMC: warning: exited " << __FUNCTION__ << " with GL errors.\n";
        abort();
    }
}

void
HPMCbuildHistopyramidFirstLevel( struct HPMCHistoPyramid* h )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        std::cerr << "HPMC: warning: entered " << __FUNCTION__ << " with GL errors.\n";
        abort();
    }

    if( h->m_base_size_l2 < 1 ) {
        return;
    }

    // first reduction
    glUseProgram( h->m_first_reduction_p );
    glActiveTextureARB( GL_TEXTURE0_ARB );
    glBindTexture( GL_TEXTURE_2D, h->m_histopyramid_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    glUniform2f( h->m_first_reduction_delta_loc,
                 -0.5/(1<<h->m_base_size_l2),
                 0.5/(1<<h->m_base_size_l2) );

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid_fbos[1] );
    glViewport( 0, 0, 1<<(h->m_base_size_l2-1), 1<<(h->m_base_size_l2-1) );

    HPMCrenderGPGPUQuad( h );

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        std::cerr << "HPMC: warning: exited " << __FUNCTION__ << " with GL errors.\n";
        abort();
    }
}

void
HPMCbuildHistopyramidUpperLevels( struct HPMCHistoPyramid* h )
{
     if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
     // rest of reductions
    glUseProgram( h->m_reduction_p );
    for(GLsizei m=2; m<=h->m_base_size_l2; m++) {
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, m-1 );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m-1 );
        glUniform2f( h->m_first_reduction_delta_loc,
                     -0.5/(1<<(h->m_base_size_l2+1-m)),
                     0.5/(1<<(h->m_base_size_l2+1-m)) );
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid_fbos[m] );
        glViewport( 0, 0, 1<<(h->m_base_size_l2-m), 1<<(h->m_base_size_l2-m) );
        HPMCrenderGPGPUQuad( h );
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }


}
