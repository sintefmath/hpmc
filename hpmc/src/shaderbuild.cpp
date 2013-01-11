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
    if( h == NULL ) {
        return false;
    }
    HPMCHistoPyramid::HistoPyramidBuild& hpb = h->m_hp_build;
    HPMCHistoPyramid::HistoPyramidBuild::BaseConstruction& base = hpb.m_base;
    HPMCHistoPyramid::HistoPyramidBuild::FirstReduction& first = hpb.m_first;
    HPMCHistoPyramid::HistoPyramidBuild::UpperReduction& upper = hpb.m_upper;

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC warning: buildHPBuildShaders called with GL errors." << endl;
#endif
        return false;
    }

    // --- build base level construction shader --------------------------------
    hpb.m_gpgpu_vertex_shader = HPMCcompileShader( HPMCgenerateDefines( h ) +
                                                   HPMCgenerateGPGPUVertexPassThroughShader( h ),
                                                   GL_VERTEX_SHADER );
    if( hpb.m_gpgpu_vertex_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build base level gpgpu vertex shader." << endl;
#endif
        return false;
    }

    // --- build base level construction shader --------------------------------
    base.m_fragment_shader = HPMCcompileShader( HPMCgenerateDefines( h ) +
                                                HPMCgenerateScalarFieldFetch( h ) +
                                                HPMCgenerateBaselevelShader( h ),
                                                GL_FRAGMENT_SHADER );
    if( base.m_fragment_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build base level construction fragment shader." << endl;
#endif
        return false;
    }

    base.m_program = glCreateProgram();
    glAttachShader( base.m_program, hpb.m_gpgpu_vertex_shader );
    glAttachShader( base.m_program, base.m_fragment_shader );
    if(! HPMClinkProgram( base.m_program ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to link base level construction program." << endl;
#endif
        return false;
    }

    // --- configure base level construction program ---------------------------
    glUseProgram( base.m_program );
    base.m_loc_threshold = HPMCgetUniformLocation( base.m_program, "HPMC_threshold" );
    GLint loc_vertex_count = HPMCgetUniformLocation( base.m_program, "HPMC_vertex_count" );
    if( loc_vertex_count != -1 ) {
        glUniform1i( loc_vertex_count, hpb.m_tex_unit_1 );
    }
    else {
#ifdef DEBUG
        cerr << "HPMC error: Failed to locate vertex count texture uniform in base level construction program." << endl;
#endif
        return false;
    }

    if( h->m_fetch.m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
        GLint loc_field = HPMCgetUniformLocation( base.m_program, "HPMC_scalarfield" );
        if( loc_field != -1 ) {
            glUniform1i( loc_field, hpb.m_tex_unit_2 );
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
    first.m_fragment_shader = HPMCcompileShader( HPMCgenerateDefines( h ) +
                                                 HPMCgenerateReductionShader( h, "floor" ),
                                                 GL_FRAGMENT_SHADER );
    if( first.m_fragment_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build first reduction fragment shader." << endl;
#endif
        return false;
    }
    first.m_program = glCreateProgram();
    glAttachShader( first.m_program, hpb.m_gpgpu_vertex_shader );
    glAttachShader( first.m_program, first.m_fragment_shader );
    if(! HPMClinkProgram( first.m_program ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to link first reduction program." << endl;
#endif
        return false;
    }

    // --- configure first pure reduction pass program -------------------------
    glUseProgram( first.m_program );
    first.m_loc_src_level = glGetUniformLocation( first.m_program, "HPMC_src_level" );
    first.m_loc_delta = glGetUniformLocation( first.m_program, "HPMC_delta" );
    GLint fr_hp_loc = HPMCgetUniformLocation( first.m_program, "HPMC_histopyramid" );
    glUniform1i( fr_hp_loc, hpb.m_tex_unit_1 );

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: GL errors building first reduction program." << endl;
#endif
        return false;
    }

    // --- build upper levels reduction pass program ---------------------------
    upper.m_fragment_shader =  HPMCcompileShader( HPMCgenerateDefines( h ) +
                                                  HPMCgenerateReductionShader( h ),
                                                  GL_FRAGMENT_SHADER );
    if( upper.m_fragment_shader == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to build upper levels reduction fragment shader." << endl;
#endif
        return false;
    }

    upper.m_program = glCreateProgram();
    glAttachShader( upper.m_program, hpb.m_gpgpu_vertex_shader );
    glAttachShader( upper.m_program, upper.m_fragment_shader );
    if(! HPMClinkProgram( upper.m_program ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to link upper levels reduction program." << endl;
#endif
        return false;
    }

    // --- configure upper levels reduction pass program -----------------------
    glUseProgram( h->m_hp_build.m_upper.m_program );
    upper.m_loc_delta = glGetUniformLocation( upper.m_program, "HPMC_delta" );
    upper.m_loc_src_level = glGetUniformLocation( upper.m_program, "HPMC_src_level" );
    GLint ur_hp_loc = HPMCgetUniformLocation( upper.m_program, "HPMC_histopyramid" );
    if( ur_hp_loc != -1 ) {
        glUniform1i( ur_hp_loc, hpb.m_tex_unit_1 );
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
