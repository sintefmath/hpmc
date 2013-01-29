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

namespace cuhpmc {

struct hp5_buildup_level_single_args
{
    uint4* __restrict__         hp_b_d;
    uint* __restrict__          sb_b_d;
    const uint* __restrict__    sb_a_d;
    uint                        N_b;
};

__global__
void
__launch_bounds__( 160 )
hp5_buildup_level_single( hp5_buildup_level_single_args a /*uint4* __restrict__  d_hp_0,
                          uint*  __restrict__  d_sb_0,
                          uint*  __restrict__  d_sb_1,
                          const uint           N_l*/ )
{
    __shared__ uint sb[160];
    const uint tid = threadIdx.x;
    const uint gid5 = 160*blockIdx.x + threadIdx.x;
    sb[threadIdx.x] = gid5 < a.N_b  ? a.sb_a_d[ gid5 ] : 0;
    __syncthreads();
    if( threadIdx.x < 32 ) {
        uint4 sums = make_uint4( sb[5*tid+0], sb[5*tid+1], sb[5*tid+2], sb[5*tid+3] );
        uint sum = sums.x + sums.y + sums.z + sums.w + sb[ 5*tid + 4 ];
        a.hp_b_d[ 32*blockIdx.x + threadIdx.x ] = sums;
        a.sb_b_d[ 32*blockIdx.x + threadIdx.x ] = sum;
    }
}

void
run_hp5_buildup_level_single( uint4*        hp_b_d,
                              uint*         sb_b_d,
                              const uint*   sb_a_d,
                              const uint    N_b,
                              cudaStream_t  stream )
{
    uint gs = (N_b+159)/160;
    uint bs = 160;

    hp5_buildup_level_single_args args;
    args.hp_b_d = hp_b_d;
    args.sb_b_d = sb_b_d;
    args.sb_a_d = sb_a_d;
    args.N_b    = N_b;

    hp5_buildup_level_single<<<gs,bs,0,stream>>>( args );

    /*
    hp5_buildup_level_single
            <<< (hp5_level_sizes[i-1]+159)/160, 160 >>>
            ( hp_d + hp5_offsets[ i-1 ],
              hp5_sb_d + hp5_offsets[ i-1 ],
              hp5_sb_d + hp5_offsets[ i   ],
              hp5_level_sizes[i-1] );
 */
}



} // of namespace cuhpmc
