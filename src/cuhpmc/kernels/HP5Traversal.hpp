#pragma once
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

// constant mem size: 64kb, cache working set: 8kb.
// Count + pad :  1+3 elements :    16 bytes :    16 bytes
// Level 0     :    4 elements :    16 bytes :    32 bytes
// Level 1     :   20 elements :    80 bytes :   112 bytes
// Level 2     :  100 elements :   400 bytes :   512 bytes
// Level 3     :  500 elements :  2000 bytes :  2112 bytes
// Level 4     : 2500 elements : 10000 bytes : 12112 bytes
// Levels 0-2: 32*4*4=512 bytes :
// Level  3:


static
__device__
__forceinline__
uint
__bitfieldextract( uint src, uint offset, uint bits )
{
    uint ret;
    asm volatile( "bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(src), "r"(offset), "r"(bits) );
    return ret;
}


static __constant__ uint  triangle_hp5_offsets[32];

static
__device__
__inline__
void
trianglePyramidDownTraverse( uint& pos,
                             uint& key,
                             const uint max_level,
                             const uint4* hp5_d )
{
    pos = 0;
    for(int l=0; l<max_level-3; l++) {
        // stored as 4 x 32 = 128 bitsc
        uint4 val = hp5_d[ triangle_hp5_offsets[l] + pos ];
        pos *= 5;
        if( val.x <= key ) {
            pos++;
            key -=val.x;
            if( val.y <= key ) {
                pos++;
                key-=val.y;
                if( val.z <= key ) {
                    pos++;
                    key-=val.z;
                    if( val.w <= key ) {
                        pos++;
                        key-=val.w;
                    }
                }
            }
        }
    }
    for(int l=max_level-3; l<max_level-1; l++) {
        // stored as 4 x 8 = 32 bits
        uint val = ((uint*)(hp5_d + triangle_hp5_offsets[ l ]))[pos];
        pos = 5*pos;
        uint t = __bitfieldextract( val, 0, 8 );
        if( t <= key ) {
            pos++;
            key -= t;
            uint t = __bitfieldextract( val, 8, 8 );
            if( t <= key ) {
                pos++;
                key -= t;
                uint t = __bitfieldextract( val, 16, 8 );
                if( t <= key ) {
                    pos++;
                    key -= t;
                    uint t = __bitfieldextract( val, 24, 8 );
                    if( t <= key ) {
                        pos++;
                        key -= t;
                    }
                }
            }
        }
    }
    {
        // stored as 4 x 4 = 16 bits
        uint val = ((unsigned short int*)(hp5_d + triangle_hp5_offsets[ max_level-1 ] ))[ pos ];
        pos = 5*pos;
        uint t = __bitfieldextract( val, 0, 4 );
        if( t <= key ) {
            pos++;
            key -= t;
            t = __bitfieldextract( val, 4, 4 );
            if( t <= key ) {
                pos++;
                key -= t;
                t = __bitfieldextract( val, 8, 4 );
                if( t <= key ) {
                    pos++;
                    key -= t;
                    t = __bitfieldextract( val, 12, 4 );
                    if( t <= key ) {
                        pos++;
                        key -= t;
                    }
                }
            }
        }
    }
}


static __constant__ uint  vertex_hp5_offsets[32];

static
__device__
__inline__
uint
vertexPyramidUpTraverse( uint adjusted_pos,
                         const uint max_level,
                         const uint4* vertex_pyramid_d )
{
    uint index = 0u;
    {   // stored as 4 x 2 = 8 bits
        uint component = adjusted_pos % 5;
        adjusted_pos = adjusted_pos/5;
        unsigned char val_ = ((unsigned char*)(vertex_pyramid_d + vertex_hp5_offsets[ max_level-1 ] ))[ adjusted_pos ];

        if( component > 3 ) { index += __bitfieldextract(val_, 6, 2); }
        if( component > 2 ) { index += __bitfieldextract(val_, 4, 2); }
        if( component > 1 ) { index += __bitfieldextract(val_, 2, 2); }
        if( component > 0 ) { index += __bitfieldextract(val_, 0, 2); }
    }
    {   // stored as 4 x 4 = 16 bits
        uint component = adjusted_pos % 5;
        adjusted_pos = adjusted_pos/5;
        uint val_ = ((unsigned short int*)(vertex_pyramid_d + vertex_hp5_offsets[ max_level-2 ] ))[ adjusted_pos ];
        if( component > 3 ) { index += __bitfieldextract(val_,12, 4); }
        if( component > 2 ) { index += __bitfieldextract(val_, 8, 4); }
        if( component > 1 ) { index += __bitfieldextract(val_, 4, 4); }
        if( component > 0 ) { index += __bitfieldextract(val_, 0, 4); }
    }
    {   // stored as 4 x 8 = 32 bits
        uint component = adjusted_pos % 5;
        adjusted_pos = adjusted_pos/5;
        uchar4 val_ = ((uchar4*)(vertex_pyramid_d + vertex_hp5_offsets[ max_level-3 ]))[adjusted_pos];
        uint4 val = make_uint4( val_.x,
                                val_.y,
                                val_.z,
                                val_.w );
        if( component > 3 ) { index += val.w; }
        if( component > 2 ) { index += val.z; }
        if( component > 1 ) { index += val.y; }
        if( component > 0 ) { index += val.x; }
    }
    for(int l=max_level-4; l>=0; l--) {
        uint component = adjusted_pos % 5;
        adjusted_pos = adjusted_pos/5;
        uint4 val = vertex_pyramid_d[ vertex_hp5_offsets[l] + adjusted_pos ];
        if( component > 3 ) { index += val.w; }
        if( component > 2 ) { index += val.z; }
        if( component > 1 ) { index += val.y; }
        if( component > 0 ) { index += val.x; }
    }
    return index;
}


} // of namespace cuhpmc
