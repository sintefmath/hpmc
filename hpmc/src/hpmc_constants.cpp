/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc_constants.cpp
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
#include <string>
#include <hpmc.h>
#include <hpmc_internal.h>

struct HPMCConstants*
HPMCcreateSingleton()
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
      std::cerr << "HPMC: createSingleton invoked with GL error.\n";
#endif
        return NULL;
    }
    struct HPMCConstants *s = new HPMCConstants;

    s->m_enumerate_vbo = 0;
    s->m_enumerate_vbo_n = 3*100000;
    s->m_edge_decode_tex = 0;
    s->m_vertex_count_tex = 0;
    s->m_gpgpu_quad_vbo = 0;

    s->m_enumerate_vbo = HPMCbuildEnumerateVBO( s->m_enumerate_vbo_n );
    if( s->m_enumerate_vbo == 0u ) {
        HPMCdestroySingleton( s );
        return NULL;
    }

    s->m_edge_decode_tex = HPMCbuildEdgeDecodeTable( );
    if( s->m_edge_decode_tex == 0u ) {
        HPMCdestroySingleton( s );
        return NULL;
    }

    s->m_vertex_count_tex = HPMCbuildVertexCountTable( );
    if( s->m_vertex_count_tex == 0u ) {
        HPMCdestroySingleton( s );
        return NULL;
    }

    s->m_gpgpu_quad_vbo = HPMCbuildGPGPUQuadVBO( );
    if( s->m_gpgpu_quad_vbo == 0u ) {
        HPMCdestroySingleton( s );
        return NULL;
    }
   if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
      std::cerr << "HPMC: createSingleton introduced with GL error.\n";
#endif
    }
    return s;
}

void
HPMCdestroySingleton( struct HPMCConstants* s )
{
    if( s == NULL ) {
#ifdef DEBUG
        std::cerr << "HPMC: destroySingleton invoked with NULL pointer.\n";
#endif
        return;
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
      std::cerr << "HPMC: destroySingleton invoked with GL error.\n";
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
      std::cerr << "HPMC: destroySingleton introduced GL error.\n";
#endif
    }
    delete s;
}
