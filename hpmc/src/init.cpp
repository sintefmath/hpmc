/* -*- mode: C++; tab-width:4; c-basic-offset: 4; indent-tabs-mode:nil -*-
 ***********************************************************************
 *
 *  File: init.cpp
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
using std::max;
using std::cerr;
using std::endl;

#ifdef _WIN32
#define log2f(x) (logf(x)*1.4426950408889634f)
#endif

// -----------------------------------------------------------------------------
bool
HPMCsetup( struct HPMCHistoPyramid* h )
{
    // not tainted, nothing to do.
    if( !h->m_tainted ) {
        return true;
    }
    if( !HPMCdetermineLayout(h) ) {
        return false;
    }
    if( !HPMCsetupTexAndFBOs(h) ) {
        return false;
    }
    if( !HPMCfreeHPBuildShaders( h ) ) {
        return false;
    }
    if( !HPMCbuildHPBuildShaders( h ) ) {
        return false;
    }
    h->m_tainted = false;
    return true;
}

// -----------------------------------------------------------------------------
bool
HPMCdetermineLayout( struct HPMCHistoPyramid* h )
{
    // --- sanity checks -------------------------------------------------------
#ifdef DEBUG
    cerr << "HPMC info: m_field.m_size = ["
         << h->m_field.m_size[0] << "x"
         << h->m_field.m_size[1] << "x"
         << h->m_field.m_size[2] << "]." << endl;
#endif
    if( (h->m_field.m_size[0] < 2) ||
        (h->m_field.m_size[1] < 2) ||
        (h->m_field.m_size[2] < 2) ||
        (16384 < h->m_field.m_size[0]) ||
        (16384 < h->m_field.m_size[1]) ||
        (16384 < h->m_field.m_size[2]) )
    {
        return false;
    }

#ifdef DEBUG
    cerr << "HPMC info: m_field.m_cells = ["
         << h->m_field.m_cells[0] << "x"
         << h->m_field.m_cells[1] << "x"
         << h->m_field.m_cells[2] << "]." << endl;
#endif
    if( (h->m_field.m_size[0] <= h->m_field.m_cells[0] ) ||
        (h->m_field.m_size[1] <= h->m_field.m_cells[1] ) ||
        (h->m_field.m_size[2] <= h->m_field.m_cells[2] ) )
    {
        return false;
    }

    // --- determine tiling ----------------------------------------------------
    h->m_tiling.m_tile_size[0] =
            1u<<(GLsizei)ceilf( log2f(
                    static_cast<float>(h->m_field.m_cells[0])/2.0f ) );
    h->m_tiling.m_tile_size[1] =
            1u<<(GLsizei)ceilf( log2f(
                    static_cast<float>(h->m_field.m_cells[1])/2.0f ) );
    float aspect =
            static_cast<float>(h->m_tiling.m_tile_size[0]) /
            static_cast<float>(h->m_tiling.m_tile_size[1]);

    h->m_tiling.m_layout[0] =
            1u<<(GLsizei)max( 0.0f,
                              ceilf( log2f( sqrt(
                                      static_cast<float>(h->m_field.m_cells[2])/aspect ) ) ) );
    h->m_tiling.m_layout[1] =
            (h->m_field.m_cells[2]+h->m_tiling.m_layout[0]-1)/h->m_tiling.m_layout[0];

    h->m_histopyramid.m_size_l2 =
            (GLsizei)ceilf( log2f(
                    static_cast<float>(
                            max( h->m_tiling.m_tile_size[0]*h->m_tiling.m_layout[0],
                                 h->m_tiling.m_tile_size[1]*h->m_tiling.m_layout[1] ) ) ) );
    h->m_histopyramid.m_size = 1<<h->m_histopyramid.m_size_l2;
    h->m_tiling.m_layout[0] = h->m_histopyramid.m_size / h->m_tiling.m_tile_size[0];
    h->m_tiling.m_layout[1] = h->m_histopyramid.m_size / h->m_tiling.m_tile_size[1];

#ifdef DEBUG
    cerr << "HPMC info: m_tiling.m_tile_size = ["
         << h->m_tiling.m_tile_size[0] << "x"
         << h->m_tiling.m_tile_size[1] << "]." << endl;
    cerr << "HPMC info: m_tiling.m_layout = ["
         << h->m_tiling.m_layout[0] << "x"
         << h->m_tiling.m_layout[1] << "]." << endl;
    cerr << "HPMC info: m_histopyramid_size_l2 = "
         << h->m_histopyramid.m_size_l2 << "." << endl;
    cerr << "HPMC info: m_histopyramid_size = "
         << h->m_histopyramid.m_size << "." << endl;
#endif

    // --- initialize vertex count to zero -------------------------------------
    h->m_histopyramid.m_top_count = 0;
    h->m_histopyramid.m_top_count_updated = true;
    return true;
}
