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
//        Start traversal in the center of the top element texel.
//        Texel shift offsets, one element per interval, updated during traversal
//        Sums for the four sub-pyramids below, from the HP tex
//        Histograms: [sum.x, sum.x+sum.y, sum.x+sum.y+sum.z, sum.x+sum.y+sum.z+sums.w]
//        Texel shift offset mask: key < hist
//              Fetch sub-pyramid sums for the four sub-pyramids.
//              Determine accummulative sums, refer to key intervals in paper.
//              hist.x = sums.x
//              hist.y = sums.x + sums.y
//              hist.z = sums.x + sums.y + sums.z
//              hist.w = sums.x + sums.y + hist.z + sums.w
//              Build a mask for next step.
//              0      <= key < hist.x  ->  mask = (1,1,1,1)
//              hist.x <= key < hist.y  ->  mask = (0,1,1,1)
//              hist.y <= key < hist.z  ->  mask = (0,0,1,1)
//              hist.z <= key < hist.w  ->  mask = (0,0,0,1)
//              Combine mask with delta_x and delta_y to shift texcoord
//              0      <= key < hist.x  ->  tp += ( -0.25, -0.25 )
//              hist.x <= key < hist.y  ->  tp += (  0.25, -0.25 )
//              hist.y <= key < hist.z  ->  tp += ( -0.25,  0.25 )
//              hist.z <= key < hist.w  ->  tp += (  0.25,  0.25 )
//          MC codes are stored in the fractional part, floor extracts the vertex count.
//          The final traversal step determines which of the four elements in the
//          baselevel that we want to descend into

void
HPMC_traverseDown( out vec2  base_texcoord,
                   out float base_value,
                   out float key_remainder,
                   in  float key_ix )
{
    vec2 texpos = vec2(0.5);
    vec4 delta_x = vec4( -0.5,  0.5, -0.5, 0.25 );
    vec4 delta_y = vec4(  0.0, -0.5,  0.0, 0.25 );
    for( int i = HPMC_HP_SIZE_L2; i>0; i-- ) {
        vec4 sums = texture2DLod( HPMC_histopyramid, texpos, float(i) );
        vec4 hist = sums;
        hist.w   += hist.z;
        hist.zw  += hist.yy;
        hist.yzw += hist.xxx;
        vec4 mask = vec4( lessThan( vec4(key_ix), hist ) );
        texpos   += vec2( dot( mask, delta_x ), dot( mask, delta_y ) );
        key_ix   -= dot( sums.xyz, vec3(1.0)-mask.xyz );
        delta_x  *= 0.5;
        delta_y  *= 0.5;
    }
    vec4 raw = texture2DLod( HPMC_histopyramid, texpos, 0.0 );
    vec4 sums = floor( raw );
    vec4 hist = sums;
    hist.w   += hist.z;
    hist.zw  += hist.yy;
    hist.yzw += hist.xxx;
    vec4 mask = vec4( lessThan( vec4(key_ix), hist ) );
    float nib = dot(vec4(mask), vec4(-1.0,-1.0,-1.0, 3.0));
    base_texcoord = texpos + vec2( dot( mask, delta_x ), dot( mask, delta_y ) );
    key_remainder = key_ix - dot( sums.xyz, vec3(1.0)-mask.xyz );
    base_value = fract( dot( raw, vec4(equal(vec4(nib),vec4(0,1,2,3))) ) );
}
