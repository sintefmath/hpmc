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
#include <cuda.h>
#include <builtin_types.h>

namespace cuhpmc {

struct hp5_buildup_apex_args
{
    uint* __restrict__          sum_d;
    uint4* __restrict__         hp_dcb_d;
    const uint* __restrict__    sb_a_d;
    uint                        N_a;
};

__global__
void
__launch_bounds__( 128 )
hp5_buildup_apex( hp5_buildup_apex_args a /*uint4* __restrict__  d_hp_012,
                  uint*  __restrict__  d_sb_3,
                  const uint           N_3*/ )
{
    __shared__ uint4 hp_012[32];
    __shared__ uint sb[128];
    const uint tid = threadIdx.x;
    sb[ tid ] = tid < a.N_a ? a.sb_a_d[ tid ] : 0 ;
    __syncthreads();
    if( tid < 25 ) {
        uint4 sums = make_uint4( sb[5*tid+0], sb[5*tid+1], sb[5*tid+2], sb[5*tid+3] );
        uint sum = sums.x+sums.y+sums.z+sums.w+sb[5*tid+4];
        hp_012[7+tid] = sums;
        sb[ tid ] = sum;
    }
    if( tid < 5 ) {
        uint4 sums = make_uint4( sb[5*tid+0], sb[5*tid+1], sb[5*tid+2], sb[5*tid+3] );
        uint sum = sums.x+sums.y+sums.z+sums.w+sb[5*tid+4];
        hp_012[2+tid] = sums;
        sb[ tid ] = sum;
    }
    if( tid < 1 ) {
        uint4 sums = make_uint4( sb[0], sb[1], sb[2],sb[3] );
        uint sum = sums.x+sums.y+sums.z+sums.w+sb[ 4 ];
        hp_012[1] = sums;
        hp_012[0] = make_uint4( sum, 1, 0, 0 );
        *a.sum_d = sum;
    }
    if( tid < 32 ) {
        a.hp_dcb_d[ tid ] = hp_012[ tid ];
    }
}

void
run_hp5_buildup_apex( uint*         sum_d,
                      uint4*        hp_dcb_d,
                      const uint*   sb_a_d,
                      const uint    N_a,
                      cudaStream_t  stream )
{
    hp5_buildup_apex_args args;
    args.sum_d      = sum_d;
    args.hp_dcb_d   = hp_dcb_d;
    args.sb_a_d     = sb_a_d;
    args.N_a        = N_a;
    hp5_buildup_apex<<<1,128,0,stream>>>( args );


    /*hp5_buildup_apex<<<1,128>>>( hp_d + 0,
                                 hp5_sb_d + 32,
                                 hp5_level_sizes[2] );
                                 */

}

} // of namespace cuhpmc
