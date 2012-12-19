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

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstdarg>
#include <hpmc.h>
#include <hpmc_internal.h>

using std::cerr;
using std::min;
using std::max;
using std::string;
using std::endl;

struct HPMCHistoPyramid*
HPMCcreateHistoPyramid2( struct HPMCConstants* constants,
                         HPMCVolumeLayout      layout,
                         ... )
{
    // --- sanity checks
    if( constants == NULL ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with NULL constants" << std::endl;
#endif
        return NULL;
    }

    if( (layout != HPMC_VOLUME_LAYOUT_FUNCTION ) &&
        (layout != HPMC_VOLUME_LAYOUT_TEXTURE_3D) &&
        (layout != HPMC_VOLUME_LAYOUT_TEXTURE_3D_PACKED) ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with illegal layout" << std::endl;
#endif
        return NULL;
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with GL errors" << std::endl;
#endif
        return NULL;
    }

    // sanity checks passed, parse configuration
    HPMCHistoPyramid* h = new HPMCHistoPyramid;
    h->m_constants = constants;
    h->m_volume_width = 0u;
    h->m_volume_height = 0u;
    h->m_volume_depth = 0u;
    h->m_volume_layout = layout;
//    h->m_volume_type = type;
    h->m_volume_type = GL_FLOAT;
    h->m_func_discrete_gradient = layout != HPMC_VOLUME_LAYOUT_FUNCTION;
    h->m_func_omit_boundary = false;
    h->m_histopyramid_tex = 0u;
    h->m_histopyramid_fbos = NULL;
    h->m_gpgpu_passthrough_v = 0u;
    h->m_baselevel_f = 0u;
    h->m_baselevel_p = 0u;
    h->m_first_reduction_f = 0u;
    h->m_first_reduction_p = 0u;
    h->m_reduction_f = 0u;
    h->m_reduction_p = 0u;
    h->m_fetch_shader = "";
    h->m_build_first_tex_unit = 0u;

    va_list ap;
    va_start(ap, layout);
    int tag = va_arg(ap, int);
    while(tag != HPMC_TAG_END ) {
        switch(tag) {
        case HPMC_TAG_WIDTH:
            h->m_volume_width = va_arg(ap, unsigned int);
            break;
        case HPMC_TAG_HEIGHT:
            h->m_volume_height = va_arg(ap, unsigned int);
            break;
        case HPMC_TAG_DEPTH:
            h->m_volume_depth = va_arg(ap, unsigned int);
            break;
        case HPMC_TAG_ELEMENT_TYPE:
            h->m_volume_type = va_arg(ap, GLenum);
            break;
        case HPMC_TAG_PRUNE_BORDER:
            h->m_func_omit_boundary = va_arg(ap, int) != 0;
            break;
        case HPMC_TAG_FETCH_SHADER:
            h->m_fetch_shader = string( va_arg(ap, const char*) );
            break;
        case HPMC_TAG_BUILD_FIRST_TEX_UNIT:
            h->m_build_first_tex_unit = va_arg(ap, unsigned int);
            break;
        default:
            goto out;
        }
        tag = va_arg(ap, int);
    }
out:
    va_end(ap);

    bool sanity = true;
    if( h->m_volume_width < 4 ) {
        cerr << "HPMC: volume width too small: " << h->m_volume_width << endl;
        sanity = false;
    }
    if( h->m_volume_height < 4 ) {
        cerr << "HPMC: volume width too small: " << h->m_volume_height << endl;
        sanity = false;
    }



    HPMCsetLayout( h );

    HPMCpushState( h );
    HPMCcreateHistopyramid( h );
    HPMCcreateShaderPrograms( h );
//    HPMCcreateTables( h );

    HPMCpopState( h );
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    return h;

}


struct HPMCHistoPyramid*
HPMCcreateHistoPyramid( struct HPMCConstants*  constants,
                        HPMCVolumeLayout       layout,
                        GLenum                 type,
                        GLsizei                volume_width,
                        GLsizei                volume_height,
                        GLsizei                volume_depth )
{
    if( constants == NULL ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with NULL constants" << std::endl;
#endif
        return NULL;
    }

    if( volume_width < 3 || volume_height < 3 || volume_depth < 3 ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with tiny volume" << std::endl;
        std::cerr << "HPMC: volume dimensions = ["
                 << volume_width << "x"
                 << volume_height << "x"
                 << volume_depth << "]"
                 << std::endl;
#endif
        return NULL;
    }

    if( type != GL_FLOAT ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with illegal type" << std::endl;
        std::cerr << "HPMC: type = " << type << std::endl;
#endif
        return NULL;
    }

    if( (layout != HPMC_VOLUME_LAYOUT_FUNCTION ) &&
        (layout != HPMC_VOLUME_LAYOUT_TEXTURE_3D) &&
        (layout != HPMC_VOLUME_LAYOUT_TEXTURE_3D_PACKED) ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with illegal layout" << std::endl;
        std::cerr << "HPMC: layout = " << layout << std::endl;
#endif
        return NULL;
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        std::cerr << "HPMC: createHistoPyramid invoked with GL errors" << std::endl;
#endif
        return NULL;
    }

    HPMCHistoPyramid* h = new HPMCHistoPyramid;
    h->m_constants = constants;
    h->m_volume_width = volume_width;
    h->m_volume_height = volume_height;
    h->m_volume_depth = volume_depth;
    h->m_volume_type = type;
    h->m_volume_layout = layout;
    h->m_func_discrete_gradient = layout != HPMC_VOLUME_LAYOUT_FUNCTION;
    h->m_func_omit_boundary = false;
    h->m_histopyramid_tex = 0u;
    h->m_histopyramid_fbos = NULL;
    h->m_gpgpu_passthrough_v = 0u;
    h->m_baselevel_f = 0u;
    h->m_baselevel_p = 0u;
    h->m_first_reduction_f = 0u;
    h->m_first_reduction_p = 0u;
    h->m_reduction_f = 0u;
    h->m_reduction_p = 0u;

    HPMCsetLayout( h );

    HPMCpushState( h );
    HPMCcreateHistopyramid( h );
    HPMCcreateShaderPrograms( h );
//    HPMCcreateTables( h );

    HPMCpopState( h );
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    return h;
}


void
HPMCbuildHistopyramidUsingTexture( struct HPMCHistoPyramid* h,
                                   GLuint             tex,
                                   GLfloat            threshold )
{
    h->m_volume_tex = tex;
    h->m_threshold = threshold;

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    HPMCpushState( h );

    HPMCbuildHistopyramidBaselevel( h );
    HPMCbuildHistopyramidFirstLevel( h );
    HPMCbuildHistopyramidUpperLevels( h );

    HPMCpopState( h );
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
}

GLuint
HPMCacquireNumberOfVertices( struct HPMCHistoPyramid* h )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    HPMCpushState( h );

    glActiveTextureARB( GL_TEXTURE0_ARB );
    glBindTexture( GL_TEXTURE_2D, h->m_histopyramid_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, h->m_base_size_l2 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, h->m_base_size_l2 );

    GLfloat mem[4];
    glGetTexImage( GL_TEXTURE_2D, h->m_base_size_l2, GL_RGBA, GL_FLOAT, &mem[0] );
    GLuint N = (GLuint)(mem[0]+mem[1]+mem[2]+mem[3]);

    HPMCpopState( h );
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    return N;
}

