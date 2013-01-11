/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: texfbosetup.cpp
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
using std::cerr;
using std::endl;

#ifdef _WIN32
#define log2f(x) (logf(x)*1.4426950408889634f)
#endif

// -----------------------------------------------------------------------------
bool
HPMCsetupTexAndFBOs( struct HPMCHistoPyramid* h )
{
    // --- if errors on state, we fail -----------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: setupTexAndFBOs called with GL errors." << endl;
#endif
        return false;
    }

    // --- create hp texture ---------------------------------------------------
    if( h->m_histopyramid.m_tex == 0 ) {
        glGenTextures( 1, &h->m_histopyramid.m_tex );
    }

    glBindTexture( GL_TEXTURE_2D, h->m_histopyramid.m_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, h->m_histopyramid.m_size_l2);
    GLsizei w = h->m_histopyramid.m_size;
    for( GLsizei i=0; i<=h->m_histopyramid.m_size_l2; i++ ) {
        glTexImage2D( GL_TEXTURE_2D, i,
                      GL_RGBA32F_ARB,
                      w, w, 0,
                      GL_RGBA, GL_FLOAT,
                      NULL );
        w = std::max(1,w/2);
    }
    //glGenerateMipmapEXT( GL_TEXTURE_2D );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    // --- create hp framebuffer objects, one fbo per level --------------------
    if( !h->m_histopyramid.m_fbos.empty() ) {
        glDeleteFramebuffersEXT( static_cast<GLsizei>( h->m_histopyramid.m_fbos.size() ),
                                 &h->m_histopyramid.m_fbos[0] );
    }
    h->m_histopyramid.m_fbos.resize( h->m_histopyramid.m_size_l2+1 );
    glGenFramebuffersEXT( static_cast<GLsizei>( h->m_histopyramid.m_fbos.size() ),
                          &h->m_histopyramid.m_fbos[0] );

    for( GLuint m=0; m<h->m_histopyramid.m_fbos.size(); m++) {
        if( h->m_constants->m_target < HPMC_TARGET_GL30_GLSL130 ) {
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, h->m_histopyramid.m_fbos[m] );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
                                   GL_COLOR_ATTACHMENT0_EXT,
                                   GL_TEXTURE_2D,
                                   h->m_histopyramid.m_tex,
                                   m );
            glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
        }
        else {
            glBindFramebuffer( GL_FRAMEBUFFER, h->m_histopyramid.m_fbos[m] );
            glFramebufferTexture2D( GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D,
                                   h->m_histopyramid.m_tex,
                                   m );
            glDrawBuffer( GL_COLOR_ATTACHMENT0 );
        }
        if( !HPMCcheckFramebufferStatus( h->m_constants, __FILE__, __LINE__ ) ) {
#ifdef DEBUG
            cerr << "HPMC error: Framebuffer for HP level " << m << " incomplete." << endl;
#endif
            return false;
        }
    }

    // --- setup pbo to for async readback of top element ----------------------
    glGenBuffers( 1, &h->m_histopyramid.m_top_pbo );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, h->m_histopyramid.m_top_pbo );
    glBufferData( GL_PIXEL_PACK_BUFFER,
                  sizeof(GLfloat)*4,
                  NULL,
                  GL_DYNAMIC_READ );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );

    // --- if we have created errors, we fail ----------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: setupTexAndFBOs produced GL errors." << endl;
#endif
        return false;
    }
    return true;
}
