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
__forceinline__
void
stepDown32( uint& pos, uint& key, const uint4& val )
{
#if 0
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
#elif 0
    pos = 5*pos;
    bool m = (val.x <= key);
    if( m ) {
        pos += 1;
        key = key - val.x;
    }
    m = m && (val.y <= key);
    if( m ) {
        pos += 1;
        key = key - val.y;
    }
    m = m && (val.z <= key);
    if( m ) {
        pos += 1;
        key = key - val.z;
    }
    m = m && (val.w <= key);
    if( m ) {
        pos += 1;
        key = key - val.w;
    }
#elif 1
    asm(
    "{"
    "    .reg .pred p;"
    "    .reg .f32 t;"
    "    .reg .u32 q;"
    "    setp.hs.u32        p, %1, %2;"
    "@p  sub.u32            %1, %1, %2;"
    "    selp.f32           t, 1.0, 0.0, p;"
    "    setp.hs.and.u32    p, %1, %3, p;"
    "@p  sub.u32            %1, %1, %3;"
    "@p  add.f32            t, t, 1.0;"
    "    setp.hs.and.u32    p, %1, %4, p;"
    "@p  sub.u32            %1, %1, %4;"
    "@p  add.f32            t, t, 1.0;"
    "    setp.hs.and.u32    p, %1, %5, p;"
    "@p  sub.u32            %1, %1, %5;"
    "@p  add.f32            t, t, 1.0;"
    "    cvt.u32.f32.rni    q, t;"
    "    mad.lo.u32         %0, 5, %0, q;"
    "}"
    : "=r"(pos), "=r"(key) : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) );
#elif 1
    asm(
    "{"
    "    .reg .pred p;"
    "    .reg .u32 v,w;"
    "    .reg .f32 t,u;"

    "    setp.hs.u32        p, %1, %2;"
    "    selp.u32           v, %2, 0, p;"
    "    selp.f32           t, 1.0, 0.0, p;"
    "    sub.u32            %1, %1, v;"

    "    setp.hs.and.u32    p, %1, %3, p;"
    "    selp.u32           v, %3, 0, p;"
    "    selp.f32           u, 1.0, 0.0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.f32            t, t, u;"

    "    setp.hs.and.u32    p, %1, %4, p;"
    "    selp.u32           v, %4, 0, p;"
    "    selp.f32           u, 1.0, 0.0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.f32            t, t, u;"

    "    setp.hs.and.u32    p, %1, %5, p;"
    "    selp.u32           v, %5, 0, p;"
    "    selp.f32           u, 1.0, 0.0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.f32            t, t, u;"
    "    cvt.u32.f32.rni    w, t;"
    "    mad.lo.u32         %0, 5, %0, w;"
    "}"
    : "=r"(pos), "=r"(key) : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) );
#elif 1
    asm(
    "{"
    "    .reg .pred p;"
    "    .reg .u32 t,u,v;"

    "    setp.hs.u32        p, %1, %2;"
    "    selp.u32           t, 1, 0, p;"
    "    selp.u32           v, %2, 0, p;"
    "    sub.u32            %1, %1, v;"

    "    setp.hs.and.u32    p, %1, %3, p;"
    "    selp.u32           u, 1, 0, p;"
    "    selp.u32           v, %3, 0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.u32            t, t, u;"

    "    setp.hs.and.u32    p, %1, %4, p;"
    "    selp.u32           u, 1, 0, p;"
    "    selp.u32           v, %4, 0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.u32            t, t, u;"

    "    setp.hs.and.u32    p, %1, %5, p;"
    "    selp.u32           u, 1, 0, p;"
    "    selp.u32           v, %5, 0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.u32            t, t, u;"
    "    mad.lo.u32         %0, 5, %0, t;"
    "}"
    : "=r"(pos), "=r"(key) : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) );
#elif 1
    asm(
    "{"
    "    .reg .pred p;"
    "    .reg .u32 t;"
    "    setp.hs.u32        p, %1, %2;"
    "@p  sub.u32            %1, %1, %2;"
    "    selp.u32           t, 1, 0, p;"
    "    setp.hs.and.u32    p, %1, %3, p;"
    "@p  sub.u32            %1, %1, %3;"
    "@p  add.u32            t, t, 1;"
    "    setp.hs.and.u32    p, %1, %4, p;"
    "@p  sub.u32            %1, %1, %4;"
    "@p  add.u32            t, t, 1;"
    "    setp.hs.and.u32    p, %1, %5, p;"
    "@p  sub.u32            %1, %1, %5;"
    "@p  add.u32            t, t, 1;"
    "    mad.lo.u32         %0, 5, %0, t;"
    "}"
    : "=r"(pos), "=r"(key) : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) );
#endif
}

static
__device__
__forceinline__
void
stepDown8( uint& pos, uint& key, const uint& val )
{
#if 0
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
#else
    uint q0, q1, q2, q3;
#if 1
    asm(
    "{  .reg .u8    x,y,z,w;"
    "   cvt.u32.u8  %0, %4;"
    "   prmt.b32    %1, %4, 0, 0x4441;"
    "   prmt.b32    %2, %4, 0, 0x4442;"
    "   prmt.b32    %3, %4, 0, 0x4443;"
    "}"
    : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "r"(val) );
#elif 1
    asm(
    "{  .reg .u8    x,y,z,w;"
    "   mov.b32     {x,y,z,w}, %4;"
    "   cvt.u32.u8  %0, x;"
    "   cvt.u32.u8  %1, y;"
    "   cvt.u32.u8  %2, z;"
    "   cvt.u32.u8  %3, w;"
    "}"
    : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "r"(val) );
#else
    asm( "bfe.u32 %0, %4,  0, 8;"
         "bfe.u32 %1, %4,  8, 8;"
         "bfe.u32 %2, %4, 16, 8;"
         "bfe.u32 %3, %4, 24, 8;"
   : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "r"(val) );
#endif
    //    uint q0 = __bitfieldextract( val, 0, 8 );
//    uint q1 = __bitfieldextract( val, 8, 8 );
//    uint q2 = __bitfieldextract( val, 16, 8 );
//    uint q3 = __bitfieldextract( val, 24, 8 );
#if 1
    pos = 5*pos;
    bool m = (q0 <= key);
    if( m ) {
        pos += 1;
        key = key - q0;
    }
    m = m && (q1 <= key);
    if( m ) {
        pos += 1;
        key = key - q1;
    }
    m = m && (q2 <= key);
    if( m ) {
        pos += 1;
        key = key - q2;
    }
    m = m && (q3 <= key );
    if( m ) {
        pos += 1;
        key = key - q3;
    }
#elif 1
    asm volatile(
    "{"
    "    .reg .pred p;"
    "    .reg .u32 t,u,v;"

    "    setp.hs.u32        p, %1, %2;"
    "    selp.u32           t, 1, 0, p;"
    "    selp.u32           v, %2, 0, p;"
    "    sub.u32            %1, %1, v;"

    "    setp.hs.and.u32    p, %1, %3, p;"
    "    selp.u32           u, 1, 0, p;"
    "    selp.u32           v, %3, 0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.u32            t, t, u;"

    "    setp.hs.and.u32    p, %1, %4, p;"
    "    selp.u32           u, 1, 0, p;"
    "    selp.u32           v, %4, 0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.u32            t, t, u;"

    "    setp.hs.and.u32    p, %1, %5, p;"
    "    selp.u32           u, 1, 0, p;"
    "    selp.u32           v, %5, 0, p;"
    "    sub.u32            %1, %1, v;"
    "    add.u32            t, t, u;"
    "    mad.lo.u32         %0, 5, %0, t;"
    "}"
    : "=r"(pos), "=r"(key) : "r"(q0), "r"(q1), "r"(q2), "r"(q3) );
#endif
#endif
}

static
__device__
__forceinline__
void
stepDown4( uint& pos, uint& key, const unsigned short int& valt )
{
#if 0
    uint val = valt;
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
#else
    uint q0, q1, q2, q3;
    asm(
    "{"
    "   .reg .u32   t;"
    "   cvt.u32.u16 t, %4;"
    //"   and.b32     %0, t, 0xf;"
    "   bfe.u32     %0, t,  0, 4;"
    "   bfe.u32     %1, t,  4, 4;"
    "   bfe.u32     %2, t,  8, 4;"
    "   bfe.u32     %3, t, 12, 4;"
    "}"
    : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "h"(valt) );
    pos = 5*pos;
    bool m = (q0 <= key);
    if( m ) {
        pos += 1;
        key = key - q0;
    }
    m = m && (q1 <= key);
    if( m ) {
        pos += 1;
        key = key - q1;
    }
    m = m && (q2 <= key);
    if( m ) {
        pos += 1;
        key = key - q2;
    }
    m = m && (q3 <= key );
    if( m ) {
        pos += 1;
        key = key - q3;
    }
#endif
}

static
__device__
__forceinline__
void
stepDown2( uint& pos, uint& key, const uint& valt )
{
#if 0
    uint4 val = make_uint4( (valt   ) & 0x3u,
                            (valt>>2) & 0x3u,
                            (valt>>4) & 0x3u,
                            (valt>>6) & 0x3u );
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
#else
    uint q0, q1, q2, q3;
    asm(
    "{"
    "   bfe.u32     %0, %4,  0, 2;"
    "   bfe.u32     %1, %4,  2, 2;"
    "   bfe.u32     %2, %4,  4, 2;"
    "   bfe.u32     %3, %4,  6, 2;"
    "}"
    : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "r"(valt) );
    pos = 5*pos;
    bool m = (q0 <= key);
    if( m ) {
        pos += 1;
        key = key - q0;
    }
    m = m && (q1 <= key);
    if( m ) {
        pos += 1;
        key = key - q1;
    }
    m = m && (q2 <= key);
    if( m ) {
        pos += 1;
        key = key - q2;
    }
    m = m && (q3 <= key );
    if( m ) {
        pos += 1;
        key = key - q3;
    }
#endif
}

static
__device__
__forceinline__
void
stepUp( uint& index, const uint& component, const uint4& val )
{
#if 1
    if( component > 3 ) { index += val.w; }
    if( component > 2 ) { index += val.z; }
    if( component > 1 ) { index += val.y; }
    if( component > 0 ) { index += val.x; }
#endif
}


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
        stepDown32( pos, key, val );
    }
    for(int l=max_level-3; l<max_level-1; l++) {
        // stored as 4 x 8 = 32 bits
        uint val = ((uint*)(hp5_d + triangle_hp5_offsets[ l ]))[pos];
        stepDown8( pos, key, val );
    }
    {   // stored as 4 x 4 = 16 bits
        unsigned short int val = ((unsigned short int*)(hp5_d + triangle_hp5_offsets[ max_level-1 ] ))[ pos ];
        stepDown4( pos, key, val );
    }
}


static __constant__ uint  vertex_hp5_offsets[32];

static
__device__
__inline__
void
vertexPyramidDownTraverse( uint& pos,
                             uint& key,
                             const uint max_level,
                             const uint4* vertex_pyramid_d )
{
    int l = 0;
    for(; l<max_level-3; l++) {
        uint4 val = vertex_pyramid_d[ vertex_hp5_offsets[l] + pos ];
        stepDown32( pos, key, val );
    }
    {   // second reduction is 4 x 8 bits = 32 bits
        uint val = ((uint*)(vertex_pyramid_d + vertex_hp5_offsets[ max_level-3 ]))[pos];
        stepDown8( pos, key, val );
    }
    {   // first reduction is 4 x 4 bits = 16 bits
        unsigned short int val = ((unsigned short int*)(vertex_pyramid_d + vertex_hp5_offsets[ max_level-2 ] ))[ pos ];
        stepDown4( pos, key, val );
    }
    {   // base layer is 4 x 2 bits = 8 bits
        unsigned char val_ = ((unsigned char*)(vertex_pyramid_d + vertex_hp5_offsets[ max_level-1 ] ))[ pos ];
        stepDown2( pos, key, val_ );
    }
}

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
