/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: triface.cpp
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

using std::endl;
using std::cerr;
using std::min;

// -----------------------------------------------------------------------------
struct HPMCTraversalHandle*
HPMCcreateTraversalHandle( struct HPMCHistoPyramid* h )
{
    if( h == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: createTraversalHandle called with h == NULL." << endl;
#endif
        return NULL;
    }

    // --- if errors on state, we fail -----------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: createTraversalHandle called with GL errors." << endl;
#endif
        return NULL;
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
    glGetIntegerv( GL_FRAMEBUFFER_BINDING,
                   reinterpret_cast<GLint*>(&old_fbo) );

    if( !HPMCsetup( h ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to untaint histopyramid." << endl;
#endif
        return NULL;
    }

    // --- restore state -------------------------------------------------------
    glBindFramebuffer( GL_FRAMEBUFFER, old_fbo );
    glUseProgram( old_prog );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );
    glPopAttrib();
    glPopClientAttrib();

    // --- if errors on state, we fail -----------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: createTraversalHandle produced GL errors." << endl;
#endif
        return NULL;
    }

    struct HPMCTraversalHandle* th = new HPMCTraversalHandle;
    th->m_handle = h;
    th->m_program = 0;
    return th;
}

// -----------------------------------------------------------------------------
void
HPMCdestroyTraversalHandle( struct HPMCTraversalHandle* th )
{
    if( th == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: destroyTraversalHandle called with th == NULL." << endl;
#endif
        return;
    }
    delete th;
}

// -----------------------------------------------------------------------------
char*
HPMCgetTraversalShaderFunctions( struct HPMCTraversalHandle* th )
{
    if( th == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: getTraversalShaderFunction called with th == NULL." << endl;
#endif
        return NULL;
    }

    // --- if errors on state, we fail -----------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: getTraversalShaderFunctions called with GL errors." << endl;
#endif
        return NULL;
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
    glGetIntegerv( GL_FRAMEBUFFER_BINDING,
                   reinterpret_cast<GLint*>(&old_fbo) );

    if( !HPMCsetup( th->m_handle ) ) {
#ifdef DEBUG
        cerr << "HPMC error: Failed to untaint histopyramid." << endl;
#endif
        return NULL;
    }

    // --- restore state -------------------------------------------------------
    glBindFramebuffer( GL_FRAMEBUFFER, old_fbo );
    glUseProgram( old_prog );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );
    glPopAttrib();
    glPopClientAttrib();

    // --- if errors on state, we fail -----------------------------------------
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: getTraversalShaderFunctions produced GL errors." << endl;
#endif
        return NULL;
    }

    // -------------------------------------------------------------------------
    std::string ret = HPMCgenerateDefines( th->m_handle )
                    + HPMCgenerateScalarFieldFetch( th->m_handle )
                    + HPMCgenerateExtractVertexFunction( th->m_handle );
    return strdup( ret.c_str() );
}

// -----------------------------------------------------------------------------
bool
HPMCsetTraversalHandleProgram( struct  HPMCTraversalHandle *th,
                               GLuint  program,
                               GLuint  tex_unit_work1,
                               GLuint  tex_unit_work2,
                               GLuint  tex_unit_work3 )
{

    // --- do all kinds of checks ----------------------------------------------
    th->m_program = 0;
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: setTraversalHandleProgram called with GL errors." << endl;
#endif
        return false;
    }
    if( th == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: passed NULL traversal handle." << endl;
#endif
        return false;
    }
    if( program == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: passed zero program." << endl;
#endif
        return false;
    }

    // --- check status etc. on program ----------------------------------------
    GLint retval;
    glGetProgramiv( program, GL_LINK_STATUS, &retval );
    if( retval != GL_TRUE ) {
#ifdef DEBUG
        cerr << "HPMC error: passed unsuccesfully linked program." << endl;
#endif
        return false;
    }
    if( tex_unit_work1 == tex_unit_work2 ) {
#ifdef DEBUG
        cerr << "HPMC error: passed identical tex unit 1 and 2." << endl;
#endif
        return false;
    }
    GLint et_loc = glGetUniformLocation( program, "HPMC_edge_table" );
    if( et_loc == -1 ) {
#ifdef DEBUG
        cerr << "HPMC error: cannot find edge table sampler uniform." << endl;
#endif
        return false;
    }
    GLint hp_loc = glGetUniformLocation( program, "HPMC_histopyramid" );
    if( hp_loc == -1 ) {
#ifdef DEBUG
        cerr << "HPMC error: cannot find histopyramid sampler uniform." << endl;
#endif
        return false;
    }

    // --- non-custom fetch checks ---------------------------------------------
    GLint sf_loc;
    if( th->m_handle->m_fetch.m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
        if( tex_unit_work1 == tex_unit_work3 ) {
#ifdef DEBUG
            cerr << "HPMC error: passed identical tex unit 1 and 3." << endl;
#endif
            return false;
        }
        if( tex_unit_work2 == tex_unit_work3 ) {
#ifdef DEBUG
            cerr << "HPMC error: passed identical tex unit 2 and 3." << endl;
#endif
            return false;
        }
        sf_loc = glGetUniformLocation( program, "HPMC_scalarfield" );
        if( sf_loc == -1 ) {
#ifdef DEBUG
            cerr << "HPMC error: cannot find scalar field uniform." << endl;
#endif
            return false;
        }
    }

    // --- get locations of uniform variables ----------------------------------
    th->m_offset_loc = glGetUniformLocation( program, "HPMC_key_offset" );
    if( th->m_offset_loc == -1 ) {
#ifdef DEBUG
        cerr << "HPMC error: cannot find key offset uniform variable." << endl;
#endif
        return false;
    }
    th->m_threshold_loc = glGetUniformLocation( program, "HPMC_threshold" );
    if( th->m_threshold_loc == -1 ) {
#ifdef DEBUG
        cerr << "HPMC error: cannot find threshold uniform variable." << endl;
#endif
        return false;
    }

    // --- store info in handle ------------------------------------------------
    th->m_program = program;
    th->m_histopyramid_unit = tex_unit_work1;
    th->m_edge_decode_unit = tex_unit_work2;
    th->m_scalarfield_unit = tex_unit_work3;

    // --- store state ---------------------------------------------------------
    GLint prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &prog );

    // --- Configure program ---------------------------------------------------
    glUseProgram( th->m_program );
    glUniform1i( et_loc, th->m_edge_decode_unit );
    glUniform1i( hp_loc, th->m_histopyramid_unit );
    if( th->m_handle->m_fetch.m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
        glUniform1i( sf_loc, th->m_scalarfield_unit );
    }

    // --- restore state -------------------------------------------------------
    glUseProgram( prog );

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: setTraversalHandleProgram produced GL errors." << endl;
#endif
        return false;
    }
    return true;
}



// -----------------------------------------------------------------------------
static bool
HPMCextractVerticesHelper( struct HPMCTraversalHandle*  th,
                           int                          transform_feedback_mode )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: extractVertices called with GL errors." << endl;
#endif
        return false;
    }
    if( th == NULL ) {
#ifdef DEBUG
        cerr << "HPMC error: passed NULL traversal handle." << endl;
#endif
        return false;
    }
    if( th->m_program == 0 ) {
#ifdef DEBUG
        cerr << "HPMC error: traversal handle has zero program." << endl;
#endif
        return false;
    }

    // --- store current state -------------------------------------------------
    GLint curr_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &curr_prog );
    glPushClientAttrib( GL_CLIENT_VERTEX_ARRAY_BIT );
    glPushAttrib( GL_TEXTURE_BIT );

    // --- retrieve number of vertices -----------------------------------------
    if( !th->m_handle->m_histopyramid.m_top_count_updated ) {
        GLfloat mem[4];
        glBindBuffer( GL_PIXEL_PACK_BUFFER,
                      th->m_handle->m_histopyramid.m_top_pbo );
        glGetBufferSubData( GL_PIXEL_PACK_BUFFER,
                            0, sizeof(GLfloat)*4,
                            &mem[0] );
        glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );

        th->m_handle->m_histopyramid.m_top_count =
                static_cast<int>( floorf(mem[0]) +
                                  floorf(mem[1]) +
                                  floorf(mem[2]) +
                                  floorf(mem[3]) );
        th->m_handle->m_histopyramid.m_top_count_updated = true;
    }

    // --- setup state ---------------------------------------------------------
    glActiveTextureARB( GL_TEXTURE0_ARB + th->m_histopyramid_unit );
    glBindTexture( GL_TEXTURE_2D, th->m_handle->m_histopyramid.m_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,
                                    th->m_handle->m_histopyramid.m_size_l2 );

    glActiveTextureARB( GL_TEXTURE0_ARB + th->m_scalarfield_unit );
    glBindTexture( GL_TEXTURE_3D, th->m_handle->m_fetch.m_tex );

    glActiveTextureARB( GL_TEXTURE0_ARB + th->m_edge_decode_unit );
    glBindTexture( GL_TEXTURE_2D, th->m_handle->m_constants->m_edge_decode_tex );

    glBindBuffer( GL_ARRAY_BUFFER, th->m_handle->m_constants->m_enumerate_vbo );
    glVertexPointer( 3, GL_FLOAT, 0, NULL );
    glEnableClientState( GL_VERTEX_ARRAY );

    glUseProgram( th->m_program );
    glUniform1f( th->m_threshold_loc, th->m_handle->m_threshold );

    // --- render triangles ----------------------------------------------------
    if( transform_feedback_mode == 1 ) {
        glBeginTransformFeedback( GL_TRIANGLES );
    }
    else if( transform_feedback_mode == 2 ) {
        glBeginTransformFeedbackNV( GL_TRIANGLES );
    }
    else if( transform_feedback_mode == 3 ) {
        glBeginTransformFeedbackEXT( GL_TRIANGLES );
    }

    GLsizei N = th->m_handle->m_histopyramid.m_top_count;
    for(GLsizei i=0; i<N; i+= th->m_handle->m_constants->m_enumerate_vbo_n) {
        glUniform1f( th->m_offset_loc, i );
        glDrawArrays( GL_TRIANGLES, 0, min( N-i,
                                            th->m_handle->m_constants->m_enumerate_vbo_n ) );
    }
    if( transform_feedback_mode == 1 ) {
        glEndTransformFeedback( );
    }
    else if( transform_feedback_mode == 2 ) {
        glEndTransformFeedbackNV( );
    }
    else if( transform_feedback_mode == 3 ) {
        glEndTransformFeedbackEXT( );
    }

    // --- restore state -------------------------------------------------------
    glPopAttrib();
    glPopClientAttrib();
    glUseProgram( curr_prog );

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
#ifdef DEBUG
        cerr << "HPMC error: extractVertices produced OpenGL errors." << endl;
#endif
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool
HPMCextractVertices( struct HPMCTraversalHandle* th )
{
    return HPMCextractVerticesHelper( th, 0 );
}

// -----------------------------------------------------------------------------
bool
HPMCextractVerticesTransformFeedback( struct HPMCTraversalHandle* th )
{
    return HPMCextractVerticesHelper( th, 1 );
}

// -----------------------------------------------------------------------------
bool
HPMCextractVerticesTransformFeedbackNV( struct HPMCTraversalHandle* th )
{
    return HPMCextractVerticesHelper( th, 2 );
}

// -----------------------------------------------------------------------------
bool
HPMCextractVerticesTransformFeedbackEXT( struct HPMCTraversalHandle* th )
{
    return HPMCextractVerticesHelper( th, 3 );
}
