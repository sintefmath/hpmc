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


// findMsb, bitCount etc. part of 400 and up.

void
HPMC_traverseDown( out vec2  base_texcoord,
                   out float base_value,
                   out float key_remainder,
                   in  float key_ix )
{

    ivec2 texpos = ivec2(0,0);

    // upper levels
    for(int i=HPMC_HP_SIZE_L2; i>0; i--) {
        vec3 sums = texelFetch( HPMC_histopyramid, texpos, i ).xyz;
        texpos = 2*texpos;
        if( sums.x <= key_ix ) {
            key_ix -= sums.x;
            if( sums.y <= key_ix ) {
                key_ix -= sums.y;
                if( sums.z <= key_ix ) {
                    key_ix -= sums.z;
                    texpos += ivec2(1,1);
                }
                else {
                    texpos += ivec2(0,1);
                }
            }
            else {
                texpos += ivec2(1,0);
            }
        }
    }

    // base level
    vec4 raw = texelFetch( HPMC_histopyramid, texpos, 0 );
    vec3 sums = floor(raw.xyz);
    texpos = 2*texpos;
    float nib;
    if( sums.x <= key_ix ) {
        key_ix -= sums.x;
        if( sums.y <= key_ix ) {
            key_ix -= sums.y;
            if( sums.z <= key_ix ) {
                key_ix -= sums.z;
                texpos += ivec2(1,1);
                nib = raw.w;
            }
            else {
                texpos += ivec2(0,1);
                nib = raw.z;
            }
        }
        else {
            texpos += ivec2(1,0);
            nib = raw.y;
        }
    }
    else {
        nib = raw.x;
    }

    // output
    base_texcoord = (0.5f/HPMC_HP_SIZE)*(vec2(texpos)+vec2(0.5));
    key_remainder = key_ix;
    base_value = fract( nib );
}
