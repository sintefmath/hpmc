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
    if( h == NULL ) {
#ifdef DEBUG
        std::cerr << "HPMC error: setupTexAndFBOs called with NULL pointer." << std::endl;
#endif
        return false;
    }
    // --- if errors on state, we fail -----------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: setupTexAndFBOs called with GL errors." << endl;
#endif
        return false;
    }
    HPMCTarget target = h->m_constants->m_target;
    HPMCHistoPyramid::HistoPyramid& hp = h->m_histopyramid;

    // --- create hp texture ---------------------------------------------------
    if( h->m_histopyramid.m_tex == 0 ) {
        glGenTextures( 1, &h->m_histopyramid.m_tex );
    }

    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glBindTexture( GL_TEXTURE_2D, hp.m_tex );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, hp.m_size_l2);
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    GLsizei w = hp.m_size;
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    for( GLsizei i=0; i<=h->m_histopyramid.m_size_l2; i++ ) {
        if(!HPMCcheckGL( __FILE__, __LINE__) ) { std::cerr << "i=" << i <<std::endl; return false; }
        if( target < HPMC_TARGET_GL30_GLSL130 ) {
            glTexImage2D( GL_TEXTURE_2D, i,
                          GL_RGBA32F_ARB,
                          w, w, 0,
                          GL_RGBA, GL_FLOAT,
                          NULL );
        }
        else {
            glTexImage2D( GL_TEXTURE_2D, i,
                          GL_RGBA32F,
                          w, w, 0,
                          GL_RGBA, GL_FLOAT,
                          NULL );
        }
        w = std::max(1,w/2);
    }
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    //glGenerateMipmapEXT( GL_TEXTURE_2D );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }

    // --- create hp framebuffer objects, one fbo per level --------------------
    if( target < HPMC_TARGET_GL30_GLSL130 ) {   // Pre GL 3.0 path
        if( !hp.m_fbos.empty() ) {
            glDeleteFramebuffersEXT( hp.m_fbos.size(), hp.m_fbos.data() );
        }
        hp.m_fbos.resize( hp.m_size_l2+1 );
        glGenFramebuffersEXT( hp.m_fbos.size(), hp.m_fbos.data() );

        for( GLuint m=0; m<hp.m_fbos.size(); m++) {
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, hp.m_fbos[m] );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                                       GL_TEXTURE_2D, hp.m_tex, m );
            glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
            GLenum status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
            if( status != GL_FRAMEBUFFER_COMPLETE_EXT ) {
#ifdef DEBUG
                std::string error;
                switch( status ) {
                case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT";
                    break;
                case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
                    error = "GL_FRAMEBUFFER_UNSUPPORTED_EXT";
                    break;
                default:
                    error = "unknown error";
                    break;
                }
                std::cerr << "HPMC error: " << error << "(" << __FILE__ << "@" << __LINE__<< ")" << std::endl;
#endif
                return false;
            }
        }
    }
    else {
        // GL 3.0 and up, doesn't use EXT_framebuffer_object
        if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
        if( !hp.m_fbos.empty() ) {
            glDeleteFramebuffers( hp.m_fbos.size(), hp.m_fbos.data() );
        }
        if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
        hp.m_fbos.resize( hp.m_size_l2+1 );
        glGenFramebuffers( hp.m_fbos.size(), hp.m_fbos.data() );
        if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }

        for( GLuint m=0; m<hp.m_fbos.size(); m++) {
            if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
            glBindFramebuffer( GL_FRAMEBUFFER, hp.m_fbos[m] );
            if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
            glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                    GL_TEXTURE_2D, hp.m_tex, m );
            if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
            glDrawBuffer( GL_COLOR_ATTACHMENT0 );
            if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
            GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
            if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
            if( status != GL_FRAMEBUFFER_COMPLETE ) {
#ifdef DEBUG
                std::string error;
                switch( status ) {
                case GL_FRAMEBUFFER_UNDEFINED:
                    error = "GL_FRAMEBUFFER_UNDEFINED";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
                    break;
                case GL_FRAMEBUFFER_UNSUPPORTED:
                    error = "GL_FRAMEBUFFER_UNSUPPORTED";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
                    error = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
                    break;
                default:
                    error = "unknown error";
                    break;
                }
                std::cerr << "HPMC error: " << error << "(" << __FILE__ << "@" << __LINE__<< ")" << std::endl;
                return false;
#endif
            }
        }
    }

    // --- setup pbo to for async readback of top element ----------------------
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glGenBuffers( 1, &h->m_histopyramid.m_top_pbo );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glBindBuffer( GL_PIXEL_PACK_BUFFER, h->m_histopyramid.m_top_pbo );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glBufferData( GL_PIXEL_PACK_BUFFER,
                  sizeof(GLfloat)*4,
                  NULL,
                  GL_DYNAMIC_READ );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }
    glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
    if(!HPMCcheckGL( __FILE__, __LINE__) ) { return false; }

    // --- if we have created errors, we fail ----------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: setupTexAndFBOs produced GL errors." << endl;
#endif
        return false;
    }
    return true;
}
