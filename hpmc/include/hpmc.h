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
/**
  * \mainpage
  *
  * \author Christopher Dyken, <christopher.dyken@sintef.no>
  *
  * \section Introduction
  *
  * HPMC is a small library that extracts iso-surfaces of volumetric data
  * directly on the GPU. It uses OpenGL to interface the GPU and assumes that
  * the volumetric data is already resident on the GPU. The output is a set of
  * vertices, optionally with normal vectors, where three and three vertices
  * form triangles of the iso-surfaces. The output can be directly extracted on
  * the fly in the vertex shader, or captured into a vertex buffer object using
  * the transform feedback extension.
  *
  * The HPMC library should work on any OpenGL-compatible hardware supporting at
  * least Shader Model 3.
  *
  * The iso-surfaces are extracted using the Marching Cubes method, implemented
  * using Histogram Pyramids, as described in "High-speed Marching Cubes using
  * Histogram Pyramids", Computer Graphics Forum 27 (8), 2008.
  *
  * \section Usage
  */
/**
  * \file hpmc.h
  *
  * \brief Library header file. Defines public interface.
  *
  * Public interface is in C, while internals are in C++.
  *
  */
/** \defgroup hpmc_public Public API
  * \{
  */
#ifndef _HPMC_H_
#define _HPMC_H_

#include <GL/glew.h>
#include <GL/gl.h>

#ifdef __cplusplus
extern "C" {
#endif

enum HPMCVolumeLayout {
    HPMC_VOLUME_LAYOUT_FUNCTION,
    HPMC_VOLUME_LAYOUT_TEXTURE_3D,
    HPMC_VOLUME_LAYOUT_TEXTURE_3D_PACKED
};

enum HPMCConfigureTag
{
    HPMC_TAG_END = 0,
    /** Width of the scalar field. */
    HPMC_TAG_WIDTH,
    /** Height of the scalar field. */
    HPMC_TAG_HEIGHT,
    /** Depth of the scalar field. */
    HPMC_TAG_DEPTH,
    HPMC_TAG_FORWARD_DIFFERENCES,
    HPMC_TAG_ELEMENT_TYPE,

    /** Skip the texels at one end of the volume.
      *
      * When using forward differences (see HPMC_TAG_FORWARD_DIFFERENCES)
      * to determine gradients (used to determine normal vectors), the texels
      * at (i+1,j,k), (i,j+1,k), and (i,j,k+1) are fetched when evaluating
      * (i,j,k). The result of this is that the layer of texels around the
      * three sides of x-max, y-max, and z-max doesn't have well-defined
      * gradients. Enabling HPMC_TAG_PRUNE_BORDER in effect reduces the
      * volume size with one pixel along each dimension.
      *
      * Note that this interact with the scaling. If using 8 texels along the
      * x-axis, texel 0 normally maps to 0.0 and texel 7 maps to 1.0. If
      * pruning is enabled, texel 6 maps to 1.0 (and texel 7 maps to
      * 1.0+(1.0/7.0)).
      *
      * Value is an integer, 0 to disable (default) or 1 to enable.
      */
    HPMC_TAG_PRUNE_BORDER,

    /** Shader source code for the fetch shader.
      *
      * The HPMC_fetch function fetches a single scalar from the scalar field.
      * \code
      * float
      * HPMC_fetch( vec3 pos );
      * \endcode
      *
      * If discrete differences is disabled, the HPMC_fetchGrad function
      * is invoked to determine the gradient (used to determine normal
      * vectors), returning the gradient in the x,y,z-components and the scalar
      * field value in the w-component.
      * \code
      * vec4
      * HPMC_fetchGrad( vec3 pos );
      * \endcode
      *
      * The x and y-coordinates of the position is scaled to fetch from texel
      * centers in [0,1] (ie., the position can be passed directly to a
      * texture lookup in a normalized texture). If sliced is enabled, the
      * z-value contains a whole number denoting the slice number (useful
      * for tiled volumes). Otherwise, the z-coordinate is scaled in the
      * same way as the x and y-coordinates.
      *
      * Value is a zero-terminated string passed as a const char*-pointer.
      * The contents of this string is copied, so the string can be safely
      * subsequently freed.
      */
    HPMC_TAG_FETCH_SHADER,

    /** First texture unit that HPMC can use running the build shader.
      *
      * Fetch functions specified with HPMC_TAG_FETCH_SHADER might want to
      * consult one or more textures, to which the application should bind
      * before running the HistoPyramid construction. Using this variable,
      * the application can allocate the first N texture units for its own
      * use.
      *
      * Value is an unsigned integer, the default value is 0.
      */
    HPMC_TAG_BUILD_FIRST_TEX_UNIT
};

struct HPMCConstants;

struct HPMCHistoPyramid;

struct HPMCTraversalHandle;

/** Creates a new singleton for the current context.
  *
  * The HPMC library requires a singleton per non-sharing context, basically,
  * one singleton per graphics card. For most applications, this amounts to a
  * single instance.
  *
  * The structure is used to hold various constant data that reside on the GPU
  * in textures and buffers.
  */
struct HPMCConstants*
HPMCcreateSingleton();

/** Destroys a singleton, freeing its resources. */
void
HPMCdestroySingleton( struct HPMCConstants* s );


/** Create a handle for a particular volume configuration.
  *
  * Creates a new handle and allocates the associated resources (like the
  * histopyramid texture, shaders, etc.)
  *
  * \param width    Width of volume configuration.
  * \param height   Height of volume configuration.
  * \param depth    Depth of volume configuration.
  * \param type     Element type used in the volume configuration.
  * \param layout   The layout used in the volume configuration.
  *
  * \return Pointer to new handle on success, NULL on failure.
  */
struct HPMCHistoPyramid*
HPMCcreateHistoPyramid( struct HPMCConstants*  s,
                        HPMCVolumeLayout       layout,
                        GLenum                 type,
                        GLsizei                volume_width,
                        GLsizei                volume_height,
                        GLsizei                volume_depth );

struct HPMCHistoPyramid*
HPMCcreateHistoPyramid2( struct HPMCConstants* s,
                         HPMCVolumeLayout      layout,
                         ... );
void
HPMCconfigureHistoPyramid( struct HPMCHistoPyramid* hp,
                           ... );


/** Free the resources associated with a handle. */
void
HPMCdestroyHandle( struct HPMCHistoPyramid* handle );

/** Builds the histopyramid using a volume texture. */
void
HPMCbuildHistopyramidUsingTexture( struct HPMCHistoPyramid* handle,
                                   GLuint             texture,
                                   GLfloat            threshold );

/** Returns the number of vertices in the histopyramid.
  *
  * \note Must be called after HPMCbuildHistopyramd*().
  */
GLuint
HPMCacquireNumberOfVertices( struct HPMCHistoPyramid* handle );


struct HPMCTraversalHandle*
HPMCcreateTraversalHandle( struct HPMCHistoPyramid* handle );

void
HPMCdestroyTraversalHandle( struct HPMCTraversalHandle* th );

char*
HPMCgetTraversalShaderFunctions( struct HPMCTraversalHandle* th );

void
HPMCinitializeTraversalHandle( struct HPMCTraversalHandle* th,
                               GLuint program,
                               GLuint scalarfield_texunit,
                               GLuint work0_tex_unit,
                               GLuint work1_tex_unit );

/** Extract triangles from the HistoPyramid
 *
 * \note Assumes that the program passed to
 * HPMCinitializeTraversalHandle is already in use on the current
 * state, as this is not set. The reason for this is to accommodate
 * geometry capture as an active transform feedback doesn't allow the
 * current program to be changed.
 */
void
HPMCextractVertices( struct   HPMCTraversalHandle* th,
                     GLsizei  triangles );



#ifdef __cplusplus
} // of extern "C"
#endif
#endif // of _HPMC_H_
/** \} */
