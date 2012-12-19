/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: constants.cpp
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
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <hpmc.h>
#include <hpmc_internal.h>

using std::vector;
using std::cerr;
using std::endl;

#define remapCode( code ) (    \
    ((((code)>>0)&0x1)<<0) |   \
    ((((code)>>1)&0x1)<<1) |   \
    ((((code)>>4)&0x1)<<2) |   \
    ((((code)>>5)&0x1)<<3) |   \
    ((((code)>>3)&0x1)<<4) |   \
    ((((code)>>2)&0x1)<<5) |   \
    ((((code)>>7)&0x1)<<6) |   \
    ((((code)>>6)&0x1)<<7) )


// -----------------------------------------------------------------------------
struct HPMCConstants*
HPMCcreateConstants()
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: createConstants called with GL errors." << endl;
#endif
        return NULL;
    }

    struct HPMCConstants *s = new HPMCConstants;
    s->m_enumerate_vbo = 0;
    s->m_edge_decode_tex = 0;
    s->m_vertex_count_tex = 0;
    s->m_gpgpu_quad_vbo = 0;

    // --- store state ---------------------------------------------------------
    glPushClientAttrib( GL_CLIENT_VERTEX_ARRAY_BIT );
    glPushAttrib( GL_TEXTURE_BIT );

    // --- build enumeration VBO, used to spawn a batch of vertices  -----------
    s->m_enumerate_vbo_n = 3*1000;
    glGenBuffers( 1, &s->m_enumerate_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, s->m_enumerate_vbo );
    glBufferData( GL_ARRAY_BUFFER,
                  3*sizeof(GLfloat)*s->m_enumerate_vbo_n,
                  NULL,
                  GL_STATIC_DRAW );
    GLfloat* ptr =
            reinterpret_cast<GLfloat*>( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );

    for(int i=0; i<s->m_enumerate_vbo_n; i++) {
        *ptr++ = static_cast<GLfloat>( i );
        *ptr++ = 0.0f;
        *ptr++ = 0.0f;
    }
    glUnmapBuffer( GL_ARRAY_BUFFER );

    // --- build edge decode table ---------------------------------------------
    vector<GLfloat> edge_decode( 256*16*4 );

    for(size_t j=0; j<256; j++ ) {
        for(size_t i=0; i<16; i++) {
            for(size_t k=0; k<4; k++)
                edge_decode[4*16*remapCode(j) + 4*i+k ]
                        = HPMC_edge_table[ HPMC_triangle_table[j][i] != -1 ? HPMC_triangle_table[j][i] : 0 ][k];
        }
    }

    glGenTextures( 1, &s->m_edge_decode_tex );
    glBindTexture( GL_TEXTURE_2D, s->m_edge_decode_tex );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_RGBA32F_ARB, 16, 256,0,
                  GL_RGBA, GL_FLOAT,
                  &edge_decode[0] );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    // --- build vertex count table --------------------------------------------
    vector<GLfloat> tricount( 256 );
    for(size_t j=0; j<256; j++) {
        size_t count;
        for(count=0; count<16; count++) {
            if( HPMC_triangle_table[j][count] == -1 ) {
                break;
            }
        }

        tricount[ remapCode(j) ] = count;
    }

    glGenTextures( 1, &s->m_vertex_count_tex );
    glBindTexture( GL_TEXTURE_1D, s->m_vertex_count_tex );
    glTexImage1D( GL_TEXTURE_1D, 0,
                  GL_ALPHA32F_ARB, 256, 0,
                  GL_ALPHA, GL_FLOAT,
                  &tricount[0] );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    // --- build GPGPU quad vbo ------------------------------------------------
    glGenBuffers( 1, &s->m_gpgpu_quad_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, s->m_gpgpu_quad_vbo );
    glBufferData( GL_ARRAY_BUFFER, sizeof(GLfloat)*3*4, &HPMC_gpgpu_quad_vertices[0], GL_STATIC_DRAW );

    // --- restore state -------------------------------------------------------
    glPopClientAttrib();
    glPopAttrib();

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: createConstants created GL errors." << endl;
#endif
        HPMCdestroyConstants( s );
        delete s;

        return NULL;
    }
    return s;
}

// -----------------------------------------------------------------------------
void
HPMCdestroyConstants( struct HPMCConstants* s )
{
    if( s == NULL ) {
        return ;
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: destroyConstants called with GL errors." << endl;
#endif
        return;
    }

    if( s->m_enumerate_vbo != 0 ) {
        glDeleteBuffers( 1, &s->m_enumerate_vbo );
    }

    if( s->m_edge_decode_tex != 0 ) {
        glDeleteTextures( 1, &s->m_edge_decode_tex );
    }

    if( s->m_vertex_count_tex != 0 ) {
        glDeleteTextures( 1, &s->m_vertex_count_tex );
    }

    if( s->m_gpgpu_quad_vbo != 0u ) {
        glDeleteBuffers( 1, &s->m_gpgpu_quad_vbo );
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: destroyConstants introduced GL errors." << endl;
#endif
        return;
    }

    delete s;
}
