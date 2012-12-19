/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: hpmc_traversal.cpp
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

struct HPMCTraversalHandle*
HPMCcreateTraversalHandle( struct HPMCHistoPyramid* handle )
{
    struct HPMCTraversalHandle* th = new HPMCTraversalHandle;
    th->m_handle = handle;
    return th;
}

void
HPMCdestroyTraversalHandle( struct HPMCTraversalHandle* th )
{
}

char*
HPMCgetTraversalShaderFunctions( struct HPMCTraversalHandle* th )
{
    std::string ret = HPMCgenerateDefines( th->m_handle )
                    + HPMCgenerateScalarFieldFetch( th->m_handle )
                    + HPMCgenerateExtractVertexFunction( th->m_handle );
    return strdup( ret.c_str() );
}

void
HPMCinitializeTraversalHandle( struct HPMCTraversalHandle* th,
                               GLuint program,
                               GLuint scalarfield_unit,
                               GLuint work0_unit,
                               GLuint work1_unit )
{
    th->m_program = program;
    th->m_scalarfield_unit = scalarfield_unit;
    th->m_histopyramid_unit = work0_unit;
    th->m_edge_decode_unit = work1_unit;

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    HPMCpushState( th->m_handle );

    glUseProgram( th->m_program );
    glUniform1i( HPMCgetUniformLocation( th->m_program, "HPMC_scalarfield" ),
                 th->m_scalarfield_unit );
    glUniform1i( HPMCgetUniformLocation( th->m_program, "HPMC_histopyramid" ),
                 th->m_histopyramid_unit );
    glUniform1i( HPMCgetUniformLocation( th->m_program, "HPMC_edge_table" ),
                 th->m_edge_decode_unit );

    th->m_offset_loc = HPMCgetUniformLocation( th->m_program, "HPMC_key_offset" );
    th->m_threshold_loc = HPMCgetUniformLocation( th->m_program, "HPMC_threshold" );

    HPMCpopState( th->m_handle );
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
}

void
HPMCextractVertices( struct   HPMCTraversalHandle* th,
                     GLsizei  vertices )
{
    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }

#ifdef DEBUG
    GLint curr_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &curr_prog );
    if( curr_prog != th->m_program ) {
        std::cerr << "HPMCextractVertices called without traversal handle's program set as current.\n";
        return;
    }
#endif

    HPMCpushState( th->m_handle );

    //    glUseProgram( th->m_program );

    glUniform1f( th->m_offset_loc, 0 );
    glUniform1f( th->m_threshold_loc, th->m_handle->m_threshold );

    glActiveTextureARB( GL_TEXTURE0_ARB + th->m_scalarfield_unit );
    glBindTexture( GL_TEXTURE_3D, th->m_handle->m_volume_tex );

    glActiveTextureARB( GL_TEXTURE0_ARB + th->m_histopyramid_unit );
    glBindTexture( GL_TEXTURE_2D, th->m_handle->m_histopyramid_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, th->m_handle->m_base_size_l2 );

    glActiveTextureARB( GL_TEXTURE0_ARB + th->m_edge_decode_unit );
    glBindTexture( GL_TEXTURE_2D, th->m_handle->m_constants->m_edge_decode_tex );

    glBindBuffer( GL_ARRAY_BUFFER, th->m_handle->m_constants->m_enumerate_vbo );
    glVertexPointer( 3, GL_FLOAT, 0, NULL );
    glEnableClientState( GL_VERTEX_ARRAY );

    for(GLsizei i=0; i<vertices; i+= th->m_handle->m_constants->m_enumerate_vbo_n) {
        glUniform1f( th->m_offset_loc, i );
        glDrawArrays( GL_TRIANGLES, 0, min( vertices-i,
                                            th->m_handle->m_constants->m_enumerate_vbo_n ) );
    }

    if( !HPMCcheckGL( __FILE__, __LINE__ ) ) {
        abort();
    }
    HPMCpopState( th->m_handle );
    // QA: if the state is rotten to begin with, resetting state might
    // create errors. currently we just eat the errors..
    while( glGetError() != GL_NO_ERROR ) {}

}
