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

template<class T>
__device__
void
fetchFromField( uint&           bp0,                // Bit mask for slice 0
                uint&           bp1,                // Bit mask for slice 1
                uint&           bp2,                // Bit mask for slice 2
                uint&           bp3,                // Bit mask for slice 3
                uint&           bp4,                // Bit mask for slice 4
                uint&           bp5,                // Bit mask for slice 5
                const T*        field,              // Sample-adjusted field pointer
                const T*        field_end,          // Pointer to buffer end
                const size_t    field_row_pitch,
                const size_t    field_slice_pitch,
                const float     iso,
                const bool      no_check )
{
    const T* llfield = field;
    if( no_check ) {
        bp0 = (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp1 = (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp2 = (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp3 = (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp4 = (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp5 = (*llfield < iso) ? 1 : 0;
    }
    else {
        bp0 = ( llfield < field_end ) && (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp1 = ( llfield < field_end ) && (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp2 = ( llfield < field_end ) && (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp3 = ( llfield < field_end ) && (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp4 = ( llfield < field_end ) && (*llfield < iso) ? 1 : 0;
        llfield += field_slice_pitch;
        bp5 = ( llfield < field_end ) && (*llfield < iso) ? 1 : 0;
    }
}


template<class T>
struct hp5_buildup_base_triple_gb_args
{
    uint4* __restrict__             d_hp_c;
    uint*  __restrict__             d_sb_c;
    uint4* __restrict__             d_hp_b;
    uint4* __restrict__             d_hp_a;
    unsigned char* __restrict__     d_case;
    float                     iso;
    uint3                     cells;
    uint3                     chunks;
    const T* __restrict__                 field;
    const T* __restrict__           field_end;
    size_t                    field_row_pitch;
    size_t                    field_slice_pitch;
    const unsigned char*                  case_vtxcnt;
};

template<class T>
__global__
void
__launch_bounds__( 160 )
hp5_buildup_base_triple_gb( hp5_buildup_base_triple_gb_args<T> a )
{
    __shared__ uint sb[800];
    __shared__ uint sh[801];

    const uint w  = threadIdx.x / 32;                                   // warp
    const uint wt = threadIdx.x % 32;                                   // thread-in-warp
    const uint sh_i = 160*w + 5*wt;                                     //
    const uint hp_b_o = 5*32*blockIdx.x + 32*w + wt;                    //
    const uint c_lix = 5*blockIdx.x + w;                                //


    const uint3 cp = make_uint3( 31*( c_lix % a.chunks.x ) + wt,          // field pos x
                                  5*( (c_lix/a.chunks.x) % a.chunks.y ),    // field pos y
                                  5*( (c_lix/a.chunks.x) / a.chunks.y ) );  // field pos.z
    const T* lfield = a.field +                                           // Field sample pointer
                      cp.x +
                      cp.y * a.field_row_pitch +
                      cp.z * a.field_slice_pitch;

    // Check if we are in danger of sampling outside the scalar field buffer
    bool no_check = lfield +
                      32 + 5*a.field_row_pitch + 5*a.field_slice_pitch < a.field_end;

    bool xmask = cp.x < a.cells.x;
    bool znocare = cp.z+5 < a.cells.z;

    // Fetch scalar field values and determine inside-outside for 5 slices

    // bp0 = { 0, 0, 0, 0, 0, 0, 0, p_000 }
    // bp1 = { 0, 0, 0, 0, 0, 0, 0, p_001 }
    // ...
    // bp5 = { 0, 0, 0, 0, 0, 0, 0, p_005}
    uint bp0, bp1, bp2, bp3, bp4, bp5;
    fetchFromField( bp0, bp1, bp2, bp3, bp4, bp5,
                    lfield, a.field_end, a.field_row_pitch, a.field_slice_pitch,
                    a.iso,
                    no_check );

    for(uint q=0; q<5; q++) {
        // Move along y to build up masks

        // bc0 = { 0, 0, 0, 0, 0, 0, 0, p_010 }
        // bc1 = { 0, 0, 0, 0, 0, 0, 0, p_011 }
        // ...
        // bc5 = { 0, 0, 0, 0, 0, 0, 0, p_015}
        uint bc0, bc1, bc2, bc3, bc4, bc5;
        fetchFromField( bc0, bc1, bc2, bc3, bc4, bc5,
                        lfield + (q+1)*a.field_row_pitch, a.field_end, a.field_row_pitch, a.field_slice_pitch,
                        a.iso, no_check );

        // Merge
        // b0 = { 0, 0, 0, 0, 0, p_010, 0, p_000 }
        // b1 = { 0, 0, 0, 0, 0, p_011, 0, p_001 }
        // ...
        // b5 = { 0, 0, 0, 0, 0, p_015, 0, p_005}
        uint b0 = bp0 + (bc0<<2);
        uint b1 = bp1 + (bc1<<2);
        uint b2 = bp2 + (bc2<<2);
        uint b3 = bp3 + (bc3<<2);
        uint b4 = bp4 + (bc4<<2);
        uint b5 = bp5 + (bc5<<2);
        // Store for next iteration
        bp0 = bc0;
        bp1 = bc1;
        bp2 = bc2;
        bp3 = bc3;
        bp4 = bc4;
        bp5 = bc5;

        // build case
        // m0_1 = { 0, p_011, 0, p_001, 0, p_010, 0, p_000 }
        // m1_1 = { 0, p_012, 0, p_002, 0, p_011, 0, p_001 }
        // m2_1 = { 0, p_013, 0, p_003, 0, p_012, 0, p_002 }
        // m3_1 = { 0, p_014, 0, p_004, 0, p_013, 0, p_003 }
        // m4_1 = { 0, p_015, 0, p_005, 0, p_014, 0, p_004 }
        uint m0_1 = b0 + (b1<<4);
        uint m1_1 = b1 + (b2<<4);
        uint m2_1 = b2 + (b3<<4);
        uint m3_1 = b3 + (b4<<4);
        uint m4_1 = b4 + (b5<<4);
        sh[ 0*160 + threadIdx.x ] = m0_1;
        sh[ 1*160 + threadIdx.x ] = m1_1;
        sh[ 2*160 + threadIdx.x ] = m2_1;
        sh[ 3*160 + threadIdx.x ] = m3_1;
        sh[ 4*160 + threadIdx.x ] = m4_1;

        uint ix_o_1 = 160*w + 32*q + wt;

        bool ymask = cp.y+q+1 < a.cells.y;
        uint sum;

        if( xmask && ymask && wt < 31 ) { // if-test needed to avoid syncthreads??
            // m0_1 = {  p_111, p_011, p_101, p_001, p_110, p_010, p_100, p_000 }
            // m1_1 = {  p_112, p_012, p_102, p_002, p_111, p_011, p_101, p_001 }
            // m2_1 = {  p_113, p_013, p_103, p_003, p_112, p_012, p_102, p_002 }
            // m3_1 = {  p_114, p_014, p_104, p_004, p_113, p_013, p_103, p_003 }
            // m4_1 = {  p_115, p_015, p_105, p_005, p_114, p_014, p_104, p_004 }

            m0_1 += (sh[ 0*160 + threadIdx.x + 1]<<1);
            m1_1 += (sh[ 1*160 + threadIdx.x + 1]<<1);
            m2_1 += (sh[ 2*160 + threadIdx.x + 1]<<1);
            m3_1 += (sh[ 3*160 + threadIdx.x + 1]<<1);
            m4_1 += (sh[ 4*160 + threadIdx.x + 1]<<1);

            uint s0_1 = a.case_vtxcnt[ m0_1 ]; // Faster to fetch from glob. mem than tex.
            uint s1_1 = a.case_vtxcnt[ m1_1 ];
            uint s2_1 = a.case_vtxcnt[ m2_1 ];
            uint s3_1 = a.case_vtxcnt[ m3_1 ];
            uint s4_1 = a.case_vtxcnt[ m4_1 ];


            if( znocare ) {
                sum = s0_1 + s1_1 + s2_1 + s3_1 + s4_1;
            }
            else {
                sum = (cp.z+0 < a.cells.z ? s0_1 : 0) +
                      (cp.z+1 < a.cells.z ? s1_1 : 0) +
                      (cp.z+2 < a.cells.z ? s2_1 : 0) +
                      (cp.z+3 < a.cells.z ? s3_1 : 0) +
                      (cp.z+4 < a.cells.z ? s4_1 : 0);
            }
            sb[ ix_o_1 ] = sum;


            if( sum > 0 ) {
                a.d_hp_a[ 5*160*blockIdx.x + ix_o_1 ] = make_uint4( s0_1, s1_1, s2_1, s3_1 );
                a.d_case[ 5*(5*160*blockIdx.x + 160*w + 32*q + wt) + 0 ] = m0_1;
                a.d_case[ 5*(5*160*blockIdx.x + 160*w + 32*q + wt) + 1 ] = m1_1;
                a.d_case[ 5*(5*160*blockIdx.x + 160*w + 32*q + wt) + 2 ] = m2_1;
                a.d_case[ 5*(5*160*blockIdx.x + 160*w + 32*q + wt) + 3 ] = m3_1;
                a.d_case[ 5*(5*160*blockIdx.x + 160*w + 32*q + wt) + 4 ] = m4_1;
            }
        }
        else {
            sb[ ix_o_1 ] = 0;
        }
    }
    uint4 bu = make_uint4( sb[sh_i+0],  sb[sh_i+1], sb[sh_i+2], sb[sh_i+3] );
    a.d_hp_b[ hp_b_o ] = bu;
    __syncthreads();
    sh[ 32*w + wt ] = bu.x + bu.y + bu.z + bu.w + sb[ sh_i + 4 ];
    __syncthreads();
    if( w == 0 ) {
        uint4 bu = make_uint4( sh[5*wt+0], sh[5*wt+1], sh[5*wt+2], sh[5*wt+3] );
        a.d_hp_c[ 32*blockIdx.x + wt ] = bu;
        a.d_sb_c[ 32*blockIdx.x + wt ] = bu.x + bu.y + bu.z + bu.w + sh[ 5*wt + 4 ];
    }
}

void
run_hp5_buildup_base_triple_gb_ub( uint4*               hp_c_d,
                                   uint*                sb_c_d,
                                   const uint           hp2_N,
                                   uint4*               hp_b_d,
                                   uint4*               hp_a_d,
                                   unsigned char*       case_d,
                                   const float          iso,
                                   const uint3          chunks,
                                   const unsigned char* field,
                                   const uint3          field_size,
                                   const unsigned char *case_vtxcnt,
                                   cudaStream_t         stream )
{
    const uint3 cells = make_uint3( field_size.x-1, field_size.y-1, field_size.z-1 );

    hp5_buildup_base_triple_gb_args<unsigned char> args;
    args.d_hp_c             = hp_c_d;
    args.d_sb_c             = sb_c_d;
    args.d_hp_b             = hp_b_d;
    args.d_hp_a             = hp_a_d;
    args.d_case             = case_d;
    args.iso                = 256.f*iso;
    args.cells              = cells;
    args.chunks             = chunks;
    args.field              = field;
    args.field_end          = field + field_size.x*field_size.y*field_size.z;
    args.field_row_pitch    = field_size.x;
    args.field_slice_pitch  = field_size.x*field_size.y;
    args.case_vtxcnt        = case_vtxcnt;

    uint gs = (hp2_N+3999)/4000;
    uint bs = 160;
    hp5_buildup_base_triple_gb<unsigned char><<<gs,bs,0, stream >>>( args );

}

} // of namespace cuhpmc
