/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
 *
 * Author(s): Christopher Dyken, <christopher.dyken@sintef.no>
 *
 * HPMC is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * HPMC is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * HPMC.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuhpmc/IsoSurface.hpp>
#include <cuhpmc/IsoSurfaceIndexed.hpp>

namespace cuhpmc {

struct hp5_buildup_level_double_args
{
    uint4* __restrict__         hp_c_d;
    uint*  __restrict__         sb_c_d;
    uint4* __restrict__         hp_b_d;
    const uint*  __restrict__   sb_a_d;
    uint                        N_b;
};

__global__
void
__launch_bounds__( 160 )
hp5_buildup_level_double( hp5_buildup_level_double_args a )
{
    __shared__ uint sb[160];
    __shared__ uint sh[800];
    const uint w  = threadIdx.x / 32;
    const uint w5 = 5*w;
    const uint wt = threadIdx.x % 32;
    const uint wt5 = 5*wt;
    const uint b32 = 32*blockIdx.x;
    for(uint p=0; p<5; p++) {
        const uint gid5 = 5*5*b32 + (w5+p)*32 + wt;
        uint v = gid5<a.N_b ? a.sb_a_d[ gid5 ] : 0;
        sh[ 32*(w5+p) + wt ] = v;
    }
    uint4 bl = make_uint4( sh[w5*32+wt5+0], sh[w5*32+wt5+1], sh[w5*32+wt5+2], sh[w5*32+wt5+3]);
    if( w5+31 < a.N_b ) {
        a.hp_b_d[ 32*5*blockIdx.x + w*32 + wt ] = bl;
    }
    sb[ 32*w+wt ] = bl.x + bl.y + bl.z + bl.w + sh[ w5*32 + wt5 + 4 ];
    __syncthreads();
    if( w == 0 ) {
        uint4 sums = make_uint4( sb[wt5+0], sb[wt5+1], sb[wt5+2], sb[wt5+3] );
        uint sum = sums.x + sums.y + sums.z + sums.w + sb[ wt5 + 4 ];
        a.hp_c_d[ b32 + wt ] = sums;
        a.sb_c_d[ b32 + wt ] = sum;
    }
}

void
IsoSurface::invokeDoubleBuildup( uint4*         pyramid_c_d,
                                 uint*          sideband_c_d,
                                 uint4*         pyramid_b_d,
                                 const uint*    sideband_a_d,
                                 const uint     N_b,
                                 cudaStream_t   stream )
{
    uint gs = (N_b+799)/800;
    uint bs = 160;
    hp5_buildup_level_double_args args;
    args.hp_c_d = pyramid_c_d;
    args.sb_c_d = sideband_c_d;
    args.hp_b_d = pyramid_b_d;
    args.sb_a_d = sideband_a_d;
    args.N_b    = N_b;
    hp5_buildup_level_double<<<gs,bs,0,stream>>>( args );
}

void
IsoSurfaceIndexed::invokeDoubleBuildup( uint level_a, cudaStream_t stream )
{
    uint gs = (m_hp5_level_sizes[level_a-1]+799)/800;
    uint bs = 160;
    hp5_buildup_level_double_args args;
    args.hp_c_d = m_triangle_pyramid_d + m_hp5_offsets[ level_a-2 ];
    args.sb_c_d = m_triangle_sideband_d + m_hp5_offsets[level_a-2];
    args.hp_b_d = m_triangle_pyramid_d + m_hp5_offsets[level_a-1];
    args.sb_a_d = m_triangle_sideband_d + m_hp5_offsets[level_a];
    args.N_b    = m_hp5_level_sizes[level_a-1];
    hp5_buildup_level_double<<<gs,bs,0,stream>>>( args );

    args.hp_c_d = m_vertex_pyramid_d + m_hp5_offsets[ level_a-2 ];
    args.sb_c_d = m_vertex_sideband_d + m_hp5_offsets[level_a-2];
    args.hp_b_d = m_vertex_pyramid_d + m_hp5_offsets[level_a-1];
    args.sb_a_d = m_vertex_sideband_d + m_hp5_offsets[level_a];
    args.N_b    = m_hp5_level_sizes[level_a-1];
    hp5_buildup_level_double<<<gs,bs,0,stream>>>( args );

}

} // of namespace cuhpmc
