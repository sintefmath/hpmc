/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: shaderbuild.cpp
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
using std::min;
using std::max;

#ifdef _WIN32
  #define log2f(x) (logf(x)*1.4426950408889634f)
#endif

// -----------------------------------------------------------------------------
bool
HPMCfreeHPBuildShaders( struct HPMCHistoPyramid* h )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: freeHPBuildShaders called with GL errors." << endl;
#endif
        return false;
    }
    // --- base level construction ---------------------------------------------
    if( h->m_hp_build.m_base.m_program != 0 ) {
        glDeleteProgram( h->m_hp_build.m_base.m_program );
        h->m_hp_build.m_base.m_program = 0;
    }
    if( h->m_hp_build.m_base.m_fragment_shader != 0 ) {
        glDeleteShader( h->m_hp_build.m_base.m_fragment_shader );
        h->m_hp_build.m_base.m_fragment_shader = 0;
    }
    // --- first pass ----------------------------------------------------------
    if( h->m_hp_build.m_first.m_program != 0 ) {
        glDeleteProgram( h->m_hp_build.m_first.m_program );
        h->m_hp_build.m_first.m_program = 0;
    }
    if( h->m_hp_build.m_first.m_fragment_shader != 0 ) {
        glDeleteShader( h->m_hp_build.m_first.m_fragment_shader );
        h->m_hp_build.m_first.m_fragment_shader = 0;
    }
    // --- upper levels pass ---------------------------------------------------
    if( h->m_hp_build.m_upper.m_program != 0 ) {
        glDeleteProgram( h->m_hp_build.m_upper.m_program );
        h->m_hp_build.m_upper.m_program = 0;
    }
    if( h->m_hp_build.m_upper.m_fragment_shader != 0 ) {
        glDeleteShader( h->m_hp_build.m_upper.m_fragment_shader );
        h->m_hp_build.m_upper.m_fragment_shader = 0;
    }
    // --- common gpgpu vertex shader ------------------------------------------
    if( h->m_hp_build.m_gpgpu_vertex_shader != 0 ) {
        glDeleteShader( h->m_hp_build.m_gpgpu_vertex_shader );
        h->m_hp_build.m_gpgpu_vertex_shader = 0;
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: freeHPBuildShaders produced GL errors." << endl;
#endif
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool
HPMCbuildHPBuildShaders( struct HPMCHistoPyramid* h )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC warning: buildHPBuildShaders called with GL errors." << endl;
#endif
        return false;
    }

    // --- build base level construction shader --------------------------------
    h->m_hp_build.m_gpgpu_vertex_shader =
            HPMCcompileShader( HPMCgenerateDefines( h ) +
                               HPMCgenerateGPGPUVertexPassThroughShader( h ),
                               GL_VERTEX_SHADER );
    if( h->m_hp_build.m_gpgpu_vertex_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build base level gpgpu vertex shader." << endl;
#endif
        return false;
    }

    // --- build base level construction shader --------------------------------
    h->m_hp_build.m_base.m_fragment_shader =
            HPMCcompileShader( HPMCgenerateDefines( h ) +
                               HPMCgenerateScalarFieldFetch( h ) +
                               HPMCgenerateBaselevelShader( h ),
                               GL_FRAGMENT_SHADER );
    if( h->m_hp_build.m_base.m_fragment_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build base level construction fragment shader." << endl;
#endif
        return false;
    }

    h->m_hp_build.m_base.m_program = glCreateProgram();
    glAttachShader( h->m_hp_build.m_base.m_program,
                    h->m_hp_build.m_gpgpu_vertex_shader );
    glAttachShader( h->m_hp_build.m_base.m_program,
                    h->m_hp_build.m_base.m_fragment_shader );
    if(! HPMClinkProgram( h->m_hp_build.m_base.m_program ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to link base level construction program." << endl;
#endif
        return false;
    }

    // --- configure base level construction program ---------------------------
    glUseProgram( h->m_hp_build.m_base.m_program );
    h->m_hp_build.m_base.m_loc_threshold =
            HPMCgetUniformLocation( h->m_hp_build.m_base.m_program,
                                    "HPMC_threshold" );

    GLint loc_vertex_count =
            HPMCgetUniformLocation( h->m_hp_build.m_base.m_program,
                                    "HPMC_vertex_count" );
    if( loc_vertex_count != -1 ) {
        glUniform1i( loc_vertex_count, h->m_hp_build.m_tex_unit_1 );
    }
    else {
#ifdef DEBUG
        cerr << "HPMC error: Failed to locate vertex count texture uniform in base level construction program." << endl;
#endif
        return false;
    }

    if( h->m_fetch.m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
        GLint loc_field =
                HPMCgetUniformLocation( h->m_hp_build.m_base.m_program,
                                        "HPMC_scalarfield" );
        if( loc_field != -1 ) {
            glUniform1i( loc_field, h->m_hp_build.m_tex_unit_2 );
        }
        else {
#ifdef DEBUG
            cerr << "HPMC error: Failed to locate scalar field texture uniform in base level construction program." << endl;
#endif
            return false;
        }
    }
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: GL errors while configuring base level construction program." << endl;
#endif
        return false;
    }

    // --- build first pure reduction pass program -----------------------------
    h->m_hp_build.m_first.m_fragment_shader =
            HPMCcompileShader( HPMCgenerateDefines( h ) +
                               HPMCgenerateReductionShader( h, "floor" ),
                               GL_FRAGMENT_SHADER );
    if( h->m_hp_build.m_first.m_fragment_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build first reduction fragment shader." << endl;
#endif
        return false;
    }
    h->m_hp_build.m_first.m_program = glCreateProgram();
    glAttachShader( h->m_hp_build.m_first.m_program,
                    h->m_hp_build.m_gpgpu_vertex_shader );
    glAttachShader( h->m_hp_build.m_first.m_program,
                    h->m_hp_build.m_first.m_fragment_shader );
    if(! HPMClinkProgram( h->m_hp_build.m_first.m_program ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to link first reduction program." << endl;
#endif
        return false;
    }

    // --- configure first pure reduction pass program -------------------------
    glUseProgram( h->m_hp_build.m_first.m_program );
    h->m_hp_build.m_first.m_loc_delta =
            HPMCgetUniformLocation( h->m_hp_build.m_first.m_program,
                                    "HPMC_delta" );
    GLint fr_hp_loc =
            HPMCgetUniformLocation( h->m_hp_build.m_first.m_program,
                                    "HPMC_histopyramid" );
    glUniform1i( fr_hp_loc, h->m_hp_build.m_tex_unit_1 );

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: GL errors building first reduction program." << endl;
#endif
        return false;
    }

    // --- build upper levels reduction pass program ---------------------------
    h->m_hp_build.m_upper.m_fragment_shader =
            HPMCcompileShader( HPMCgenerateDefines( h ) +
                               HPMCgenerateReductionShader( h ),
                               GL_FRAGMENT_SHADER );
    if( h->m_hp_build.m_upper.m_fragment_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build upper levels reduction fragment shader." << endl;
#endif
        return false;
    }

    h->m_hp_build.m_upper.m_program = glCreateProgram();
    glAttachShader( h->m_hp_build.m_upper.m_program,
                    h->m_hp_build.m_gpgpu_vertex_shader );
    glAttachShader( h->m_hp_build.m_upper.m_program,
                    h->m_hp_build.m_upper.m_fragment_shader );
    if(! HPMClinkProgram( h->m_hp_build.m_upper.m_program ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to link upper levels reduction program." << endl;
#endif
        return false;
    }

    // --- configure upper levels reduction pass program -----------------------
    glUseProgram( h->m_hp_build.m_upper.m_program );
    h->m_hp_build.m_upper.m_loc_delta =
            HPMCgetUniformLocation( h->m_hp_build.m_upper.m_program,
                                    "HPMC_delta" );

    GLint ur_hp_loc =
            HPMCgetUniformLocation( h->m_hp_build.m_upper.m_program,
                                    "HPMC_histopyramid" );
    if( ur_hp_loc != -1 ) {
        glUniform1i( ur_hp_loc, h->m_hp_build.m_tex_unit_1 );
    }
    else {
#ifdef DEBUG
        cerr << "HPMC error: Can't find HP tex uniform in upper levels reduction program." << endl;
#endif
        return false;
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: GL errors configuring upper levels reduction program." << endl;
#endif
        return false;
    }
    return true;
}
