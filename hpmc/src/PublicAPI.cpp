/***********************************************************************
 *
 *  File: hpiface.cpp
 *
 *  Created: 2012-11-23
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
#include <algorithm>
#include <string>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <hpmc.h>
#include <hpmc_internal.h>
#include "Constants.hpp"
#include "IsoSurface.hpp"
#include "IsoSurfaceRenderer.hpp"
#include "Logger.hpp"

static const std::string package = "HPMC.publicAPI";

using namespace HPMC;

struct HPMCConstants*
HPMCcreateConstants( HPMCTarget target, HPMCDebugBehaviour debug )
{
    struct HPMCConstants *constants = NULL;
    try {
        constants = new HPMCConstants( target, debug );
        Logger log( constants, package + "createConstants", true );
        constants->init();
    }
    catch( std::runtime_error& e ) {
        std::cerr << e.what() << std::endl;
    }
    return constants;
}

void
HPMCdestroyConstants( struct HPMCConstants* constants )
{
    if( constants == NULL ) {
        return;
    }
    else {
        delete constants;
    }
}

struct HPMCIsoSurface*
HPMCcreateIsoSurface( struct HPMCConstants* constants )
{
    if( constants == NULL ) {
        return NULL;
    }

    Logger log( constants, package + "createIsoSurface", true );
    HPMCIsoSurface* h = new HPMCIsoSurface( constants );
    h->init();
    return h;
}


void
HPMCsetLatticeSize( struct HPMCIsoSurface*  h,
                    GLsizei                   x_size,
                    GLsizei                   y_size,
                    GLsizei                   z_size )
{
    Logger log( h->constants(), package + ".setLatticeSize", true );

    h->field().m_size[0] = x_size;
    h->field().m_size[1] = y_size;
    h->field().m_size[2] = z_size;
    h->field().m_cells[0] = std::max( (GLsizei)1u, h->field().m_size[0] )-(GLsizei)1u;
    h->field().m_cells[1] = std::max( (GLsizei)1u, h->field().m_size[1] )-(GLsizei)1u;
    h->field().m_cells[2] = std::max( (GLsizei)1u, h->field().m_size[2] )-(GLsizei)1u;
    h->taint();

    if( h->constants()->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "field.size = [ "
          << h->field().m_size[0] << " x "
          << h->field().m_size[1] << " x "
          << h->field().m_size[2] << " ]";
        log.debugMessage( o.str() );

        o.str("");
        o << "field.cells = [ "
          << h->field().m_cells[0] << " x "
          << h->field().m_cells[1] << " x "
          << h->field().m_cells[2] << " ]";
        log.debugMessage( o.str() );
    }
}


void
HPMCsetGridSize( struct HPMCIsoSurface*  h,
                 GLsizei                   x_size,
                 GLsizei                   y_size,
                 GLsizei                   z_size )
{
    Logger log( h->constants(), package + ".setGridSize", true );
    h->field().m_cells[0] = x_size;
    h->field().m_cells[1] = y_size;
    h->field().m_cells[2] = z_size;
    h->taint();
    if( h->constants()->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "field.cells = [ "
          << h->field().m_cells[0] << " x "
          << h->field().m_cells[1] << " x "
          << h->field().m_cells[2] << " ]";
        log.debugMessage( o.str() );
    }
}


void
HPMCsetGridExtent( struct HPMCIsoSurface*  h,
                   GLfloat                   x_extent,
                   GLfloat                   y_extent,
                   GLfloat                   z_extent )
{
    Logger log( h->constants(), package + ".setGridExtent", true );
    h->field().m_extent[0] = x_extent;
    h->field().m_extent[1] = y_extent;
    h->field().m_extent[2] = z_extent;
    h->taint();
    if( h->constants()->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "grid.extent = [ "
          << h->field().m_extent[0] << " x "
          << h->field().m_extent[1] << " x "
          << h->field().m_extent[2] << " ]";
        log.debugMessage( o.str() );
    }
}

void
HPMCsetFieldAsBinary( struct HPMCIsoSurface*  h )
{
    Logger log( h->constants(), package + ".setFieldAsBinary", true );
    h->field().m_binary = true;
    h->taint();
}


bool
HPMCsetFieldTexture3D( struct HPMCIsoSurface*  h,
                       GLuint                  texture,
                       GLenum                  field,
                       GLenum                  gradient )
{
    if( h == NULL ) {
        return false;
    }
    Logger log( h->constants(), package + ".setFieldTexture3D", true );
    if( texture == 0 ) {
        log.errorMessage( "Illegal texture name" );
        return false;
    }
    if( (field != GL_RED) && (field != GL_ALPHA) ) {
        log.errorMessage( "Illegal field enum" );
        return false;
    }
    if( (gradient != GL_RGB) && (gradient != GL_NONE) ) {
        log.errorMessage( "Illegal gradient enum" );
        return false;
    }
    if( (gradient == GL_RGB) && (field == GL_RED) ) {
        log.errorMessage( "field and gradient channels are not distinct" );
        return false;
    }

    h->field().m_tex = texture;
    log.setObjectLabel( GL_TEXTURE, h->field().m_tex, "field texture 3D" );

    if( (h->field().m_mode != HPMC_VOLUME_LAYOUT_TEXTURE_3D ) ||
        (h->field().m_tex_field_channel != field ) ||
        (h->field().m_tex_gradient_channels != gradient ) )
    {
        h->field().m_mode = HPMC_VOLUME_LAYOUT_TEXTURE_3D;
        h->field().m_tex_field_channel = field;
        h->field().m_tex_gradient_channels = gradient;
        h->m_hp_build.m_tex_unit_1 = 0;
        h->m_hp_build.m_tex_unit_2 = 1;
        h->taint();
        if( h->constants()->debugBehaviour() != HPMC_DEBUG_NONE ) {
            std::stringstream o;
            o << "field mode=TEX3D";
            switch( field ) {
            case GL_RED:
                o << ", field=RED";
                break;
            case GL_ALPHA:
                o << ", field=ALPHA";
                break;
            default:
                break;
            }
            switch( gradient ) {
            case GL_RGB:
                o << ", gradient=RGB";
                break;
            case GL_NONE:
                o << ", gradient=NONE";
                break;
            default:
                break;
            }
            o << ", build.texunits={"
              << h->m_hp_build.m_tex_unit_1 << ", "
              << h->m_hp_build.m_tex_unit_2 << " }";
            log.debugMessage( o.str() );
        }
    }
    return true;
}


void
HPMCsetFieldCustom( struct HPMCIsoSurface*  h,
                    const char*               shader_source,
                    GLuint                    builder_texunit,
                    GLboolean                 gradient )
{
    Logger log( h->constants(), package + ".setFieldCustom", true );
    h->field().m_mode = HPMC_VOLUME_LAYOUT_CUSTOM;
    h->field().m_shader_source = shader_source;
    h->field().m_tex = 0;
    h->field().m_tex_field_channel = GL_RED;
    h->field().m_tex_gradient_channels = (gradient==GL_TRUE?GL_RGB:GL_NONE);
    h->m_hp_build.m_tex_unit_1 = builder_texunit;
    h->m_hp_build.m_tex_unit_2 = builder_texunit+1;
    h->taint();
}


GLuint
HPMCgetBuilderProgram( struct HPMCIsoSurface*  h )
{
    if( h == NULL ) {
        return 0;
    }
    Logger log( h->constants(), package + ".getBuilderProgram", true );
    h->untaint();
    return h->baseLevelBuilder().program();
}


// -----------------------------------------------------------------------------
void
HPMCdestroyIsoSurface( struct HPMCIsoSurface* h )
{
    if( h == NULL ) {
        return;
    }
    Logger log( h->constants(), package + ".destroyIsoSurface" );
    delete h;
}

// -----------------------------------------------------------------------------
void
HPMCbuildIsoSurface( struct   HPMCIsoSurface* h,
                       GLfloat  threshold )
{
    if( h == NULL ) {
        return;
    }
    Logger log( h->constants(), package + ".buildIsoSurface", true );
/*
    std::cerr << __LINE__ << "\t" <<
                 th->m_handle->histoPyramid().texture() << "\n";
*/
    // --- store state ---------------------------------------------------------
    GLint old_viewport[4];
    GLuint old_pbo;
    GLuint old_fbo;
    GLuint old_prog;
    glGetIntegerv( GL_VIEWPORT, old_viewport );
    glGetIntegerv( GL_CURRENT_PROGRAM, reinterpret_cast<GLint*>(&old_prog) );
    glGetIntegerv( GL_PIXEL_PACK_BUFFER_BINDING, reinterpret_cast<GLint*>(&old_pbo) );
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&old_fbo) );

    h->setThreshold( threshold );
    h->build();

    // --- restore state -------------------------------------------------------

    if( h->constants()->target() >= HPMC_TARGET_GL30_GLSL130 ) {
        glBindVertexArray( 0 ); // GPGPU quad uses VAO.
    }
    glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );
    glBindFramebuffer( GL_FRAMEBUFFER, old_fbo );
    glViewport( old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3] );
    glUseProgram( old_prog );

    // should check for errors and h->setAsBroken()?
}

GLuint
HPMCacquireNumberOfVertices( struct HPMCIsoSurface* h )
{
    if( h == NULL ) {
        return 0;
    }
    Logger log( h->constants(), package + ".acquireNumberOfVertices", true );
    return h->vertexCount();
}


struct HPMCIsoSurfaceRenderer*
HPMCcreateIsoSurfaceRenderer( struct HPMCIsoSurface* h )
{
    if( h == NULL ) {
        return NULL;
    }
    Logger log( h->constants(), "HPMC.publicAPI.createIsoSurfaceRenderer", true );
    GLint old_viewport[4];
    GLuint old_pbo, old_fbo, old_prog;
    glGetIntegerv( GL_VIEWPORT, old_viewport );
    glGetIntegerv( GL_CURRENT_PROGRAM, reinterpret_cast<GLint*>(&old_prog) );
    glGetIntegerv( GL_PIXEL_PACK_BUFFER_BINDING, reinterpret_cast<GLint*>(&old_pbo) );
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&old_fbo) );
    if( !h->untaint() ) {
        log.errorMessage( "Failed to untaint histopyramid" );
        return NULL;
    }
    glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );
    glBindFramebuffer( GL_FRAMEBUFFER, old_fbo );
    glViewport( old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3] );
    glUseProgram( old_prog );
    return new HPMCIsoSurfaceRenderer(h);
}

// -----------------------------------------------------------------------------
void
HPMCdestroyIsoSurfaceRenderer( struct HPMCIsoSurfaceRenderer* th )
{
    if( th == NULL ) {
        return;
    }
    Logger log( th->m_handle->constants(), package + ".destroyIsoSurfaceRenderer", true );
    delete th;
}

char*
HPMCisoSurfaceRendererShaderSource( struct HPMCIsoSurfaceRenderer* th )
{
    if( th == NULL ) {
        return NULL;
    }
    Logger log( th->m_handle->constants(), package + ".createIsoSurfaceRenderer", true );
    GLint old_viewport[4];
    GLuint old_pbo, old_fbo, old_prog;
    glGetIntegerv( GL_VIEWPORT, old_viewport );
    glGetIntegerv( GL_CURRENT_PROGRAM, reinterpret_cast<GLint*>(&old_prog) );
    glGetIntegerv( GL_PIXEL_PACK_BUFFER_BINDING, reinterpret_cast<GLint*>(&old_pbo) );
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&old_fbo) );
    if( !th->m_handle->untaint() ) {
        log.errorMessage( "Failed to untaint histopyramid" );
        return NULL;
    }
    glBindBuffer( GL_PIXEL_PACK_BUFFER, old_pbo );
    glBindFramebuffer( GL_FRAMEBUFFER, old_fbo );
    glViewport( old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3] );
    glUseProgram( old_prog );

    return strdup( th->extractionSource().c_str() );
}


bool
HPMCsetIsoSurfaceRendererProgram( struct  HPMCIsoSurfaceRenderer *th,
                                  GLuint  program,
                                  GLuint  tex_unit_work1,
                                  GLuint  tex_unit_work2,
                                  GLuint  tex_unit_work3 )
{
    if( th == NULL ) {
        return false;
    }
    GLuint old_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, reinterpret_cast<GLint*>(&old_prog) );
    bool retval = th->setProgram( program, tex_unit_work1, tex_unit_work2, tex_unit_work3 );
    glUseProgram( old_prog );
    return retval;
}

// -----------------------------------------------------------------------------
bool
HPMCextractVertices( struct HPMCIsoSurfaceRenderer* th, GLboolean flip_orientation )
{
    if( th == NULL ) {
        return false;
    }
    GLint curr_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &curr_prog );
    th->draw( 0, flip_orientation==GL_TRUE );
    glUseProgram( curr_prog );
    return true;
}

// -----------------------------------------------------------------------------
bool
HPMCextractVerticesTransformFeedback( struct HPMCIsoSurfaceRenderer* th, GLboolean flip_orientation )
{
    if( th == NULL ) {
        return false;
    }
    GLint curr_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &curr_prog );
    th->draw( 1, flip_orientation==GL_TRUE );
    glUseProgram( curr_prog );
    return true;
}

// -----------------------------------------------------------------------------
bool
HPMCextractVerticesTransformFeedbackNV( struct HPMCIsoSurfaceRenderer* th, GLboolean flip_orientation )
{
    if( th == NULL ) {
        return false;
    }
    GLint curr_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &curr_prog );
    th->draw( 2, flip_orientation==GL_TRUE );
    glUseProgram( curr_prog );
    return true;
}

// -----------------------------------------------------------------------------
bool
HPMCextractVerticesTransformFeedbackEXT( struct HPMCIsoSurfaceRenderer* th, GLboolean flip_orientation )
{
    if( th == NULL ) {
        return false;
    }
    GLint curr_prog;
    glGetIntegerv( GL_CURRENT_PROGRAM, &curr_prog );
    th->draw( 3, flip_orientation==GL_TRUE );
    glUseProgram( curr_prog );
    return true;
}
