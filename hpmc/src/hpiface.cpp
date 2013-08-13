/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpiface.cpp
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

// -----------------------------------------------------------------------------
struct HPMCHistoPyramid*
HPMCcreateHistoPyramid( struct HPMCConstants* constants )
{
    if( constants == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: createHistoPyramid called with constants = NULL." << endl;
#endif
        return NULL;
    }
    HPMCHistoPyramid* h = new HPMCHistoPyramid;

    h->m_tainted = true;
    h->m_broken = true;
    h->m_constants = constants;

    h->m_tiling.m_tile_size[0] = 0;
    h->m_tiling.m_tile_size[1] = 0;
    h->m_tiling.m_layout[0] = 0;
    h->m_tiling.m_layout[1] = 0;

    h->m_histopyramid.m_size = 0;
    h->m_histopyramid.m_size_l2 = 0;
    h->m_histopyramid.m_tex = 0;
    h->m_histopyramid.m_top_pbo = 0;

    h->m_field.m_size[0] = 0;
    h->m_field.m_size[1] = 0;
    h->m_field.m_size[2] = 0;
    h->m_field.m_cells[0] = 0;
    h->m_field.m_cells[1] = 0;
    h->m_field.m_cells[2] = 0;
    h->m_field.m_extent[0] = 1.0f;
    h->m_field.m_extent[1] = 1.0f;
    h->m_field.m_extent[2] = 1.0f;
    h->m_field.m_binary = false;

    h->m_fetch.m_mode = HPMC_VOLUME_LAYOUT_TEXTURE_3D;
    h->m_fetch.m_shader_source = "";
    h->m_fetch.m_tex = 0;
    h->m_fetch.m_gradient = false;

    h->m_hp_build.m_tex_unit_1 = 0;
    h->m_hp_build.m_tex_unit_2 = 1;
    h->m_hp_build.m_gpgpu_vertex_shader = 0;
    h->m_hp_build.m_base.m_fragment_shader = 0;
    h->m_hp_build.m_base.m_program = 0;
    h->m_hp_build.m_first.m_fragment_shader = 0;
    h->m_hp_build.m_first.m_program = 0;
    h->m_hp_build.m_upper.m_fragment_shader = 0;
    h->m_hp_build.m_upper.m_program = 0;

    return h;
}

// -----------------------------------------------------------------------------
void
HPMCsetLatticeSize( struct HPMCHistoPyramid*  h,
                    GLsizei                   x_size,
                    GLsizei                   y_size,
                    GLsizei                   z_size )
{
    h->m_field.m_size[0] = x_size;
    h->m_field.m_size[1] = y_size;
    h->m_field.m_size[2] = z_size;
    h->m_field.m_cells[0] = max( (GLsizei)1u, h->m_field.m_size[0] )-(GLsizei)1u;
    h->m_field.m_cells[1] = max( (GLsizei)1u, h->m_field.m_size[1] )-(GLsizei)1u;
    h->m_field.m_cells[2] = max( (GLsizei)1u, h->m_field.m_size[2] )-(GLsizei)1u;
    h->m_tainted = true;
    h->m_broken = false;
}

// -----------------------------------------------------------------------------
void
HPMCsetGridSize( struct HPMCHistoPyramid*  h,
                 GLsizei                   x_size,
                 GLsizei                   y_size,
                 GLsizei                   z_size )
{
    h->m_field.m_cells[0] = x_size;
    h->m_field.m_cells[1] = y_size;
    h->m_field.m_cells[2] = z_size;
    h->m_tainted = true;
    h->m_broken = false;
}

// -----------------------------------------------------------------------------
void
HPMCsetFieldAsBinary( struct HPMCHistoPyramid* h )
{
    h->m_field.m_binary = true;
}

// -----------------------------------------------------------------------------
void
HPMCsetFieldAsContinuous( struct HPMCHistoPyramid* h )
{
    h->m_field.m_binary = false;    
}

// -----------------------------------------------------------------------------
void
HPMCsetGridExtent( struct HPMCHistoPyramid*  h,
                   GLfloat                   x_extent,
                   GLfloat                   y_extent,
                   GLfloat                   z_extent )
{
    h->m_field.m_extent[0] = x_extent;
    h->m_field.m_extent[1] = y_extent;
    h->m_field.m_extent[2] = z_extent;
    h->m_tainted = true;
    h->m_broken = false;
#ifdef DEBUG
    cerr << "HPMC info: grid extent x = " << h->m_field.m_extent[0] << endl;
    cerr << "HPMC info: grid extent y = " << h->m_field.m_extent[1] << endl;
    cerr << "HPMC info: grid extent z = " << h->m_field.m_extent[2] << endl;
#endif
}

// -----------------------------------------------------------------------------
void
HPMCsetFieldTexture3D( struct HPMCHistoPyramid*  h,
                       GLuint                    texture,
                       GLboolean                 gradient )
{
    h->m_fetch.m_tex = texture;

    bool grad = ( gradient==GL_TRUE? true : false );

    // significant changes trigger a rebuild of everything
    if( (h->m_fetch.m_mode != HPMC_VOLUME_LAYOUT_TEXTURE_3D) ||
        (h->m_fetch.m_gradient != grad) )
    {
        h->m_fetch.m_mode = HPMC_VOLUME_LAYOUT_TEXTURE_3D;
        h->m_fetch.m_gradient = grad;
        h->m_hp_build.m_tex_unit_1 = 0;
        h->m_hp_build.m_tex_unit_2 = 1;
        h->m_tainted = true;
        h->m_broken = false;
    }
}

// -----------------------------------------------------------------------------
void
HPMCsetFieldCustom( struct HPMCHistoPyramid*  h,
                    const char*               shader_source,
                    GLuint                    builder_texunit,
                    GLboolean                 gradient )
{
    h->m_fetch.m_mode = HPMC_VOLUME_LAYOUT_CUSTOM;
    h->m_fetch.m_shader_source = shader_source;
    h->m_fetch.m_gradient = ( gradient==GL_TRUE? true : false );
    h->m_hp_build.m_tex_unit_1 = builder_texunit;
    h->m_hp_build.m_tex_unit_2 = builder_texunit+1;
    h->m_tainted = true;
    h->m_broken = false;
}

// -----------------------------------------------------------------------------
GLuint
HPMCgetBuilderProgram( struct HPMCHistoPyramid*  h )
{
    if( h == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: h == NULL." << endl;
#endif
        return 0;
    }
    if( h->m_broken ) {
#ifdef DEBUG
        cerr << "HPMC error: h is broken." << endl;
#endif
        return 0;
    }
    if( h->m_tainted ) {
        HPMCsetup( h );
    }
    return h->m_hp_build.m_base.m_program;
}

// -----------------------------------------------------------------------------
void
HPMCbuildHistopyramid( struct   HPMCHistoPyramid* h,
                       GLfloat  threshold )
{
    if( h == NULL || h->m_broken ) {
        return;
    }
    // -------------------------------------------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: buildHistopyramid called with errors on state." << endl;
#endif
        return;
    }

    // --- store state ---------------------------------------------------------
    glPushClientAttrib( GL_CLIENT_VERTEX_ARRAY_BIT );
    glPushAttrib( GL_VIEWPORT_BIT | GL_TEXTURE_BIT );
    GLuint old_pbo;
    GLuint old_prog;
    GLuint old_fbo;
    glGetIntegerv( GL_PIXEL_PACK_BUFFER_BINDING,
                   reinterpret_cast<GLint*>(&old_pbo) );
    glGetIntegerv( GL_CURRENT_PROGRAM,
                   reinterpret_cast<GLint*>(&old_prog) );
    if( h->m_constants->m_target < HPMC_TARGET_GL30_GLSL130 ) {
        glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, reinterpret_cast<GLint*>(&old_fbo) );
    }
    else {
        glGetIntegerv( GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&old_fbo) );
    }

    // --- if HP is reconfigured, setup shaders and fbo's ----------------------
    if( h->m_tainted ) {
        HPMCsetup( h );
    }

    // --- if everything is O.K., do construction pass -------------------------
    if(!h->m_tainted ) {
        h->m_threshold = threshold;
        if(! HPMCtriggerHistopyramidBuildPasses( h ) ) {
            h->m_broken = true;
        }
    }

    // --- restore state -------------------------------------------------------
    if( h->m_constants->m_target < HPMC_TARGET_GL30_GLSL130 ) {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, old_fbo );
    }
    else {
        glBindFramebuffer( GL_FRAMEBUFFER, old_fbo );
    }
    glUseProgram( old_prog );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );
    glPopAttrib();
    glPopClientAttrib();

    // -------------------------------------------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: buildHistopyramid produced GL errors." << endl;
#endif
        h->m_broken = true;
        return;
    }
}

// -----------------------------------------------------------------------------
GLuint
HPMCacquireNumberOfVertices( struct HPMCHistoPyramid* h )
{
    if( h == NULL || h->m_broken ) {
        return 0;
    }

    // --- retrieve number of vertices -----------------------------------------
    if( !h->m_histopyramid.m_top_count_updated ) {

        // ---------------------------------------------------------------------
        if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
            cerr << "HPMC error: acquireNumberOfVertices called with GL errors." << endl;
#endif
            return 0;
        }

        // --- store state -----------------------------------------------------
        GLuint old_pbo;
        glGetIntegerv( GL_PIXEL_PACK_BUFFER_BINDING,
                       reinterpret_cast<GLint*>(&old_pbo) );

        // --- read values in fbo (forcing a sync) -----------------------------
        GLfloat mem[4];
        glBindBuffer( GL_PIXEL_PACK_BUFFER,
                      h->m_histopyramid.m_top_pbo );
        glGetBufferSubData( GL_PIXEL_PACK_BUFFER,
                            0, sizeof(GLfloat)*4,
                            &mem[0] );
        h->m_histopyramid.m_top_count =
                static_cast<int>( floorf(mem[0]) +
                                  floorf(mem[1]) +
                                  floorf(mem[2]) +
                                  floorf(mem[3]) );
        h->m_histopyramid.m_top_count_updated = true;

        // --- restore state ---------------------------------------------------
        glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );


        // ---------------------------------------------------------------------
        if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
            cerr << "HPMC error: acquireNumberOfVertices produced GL errors." << endl;
#endif
            h->m_broken = true;
            return 0;
        }
    }

    return h->m_histopyramid.m_top_count;
}

