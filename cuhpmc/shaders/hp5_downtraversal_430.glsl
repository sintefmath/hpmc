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
layout(binding=2)   uniform usamplerBuffer  hp5_tex;
                    uniform uint            hp5_offsets[ CUHPMC_LEVELS ];

#ifdef CUHMPC_CONF_CONSTMEM_APEX
layout(std140)      uniform HP5Apex {
    uvec4    hp5_apex[528];
};
#endif

void
hp5_downtraverse( out uint pos, out uint key_remainder, in uint key )
{
    uint p = 0;

    uint l = 0;
#ifdef CUHMPC_CONF_CONSTMEM_APEX
    // Use bindable uniform (=constmem) for apex. Layout is known here, so we
    // can directly calculate the offsets.
    uint o = 1;
    uint a = 1;
    for(; l<4; l++) {
        uvec4 val = hp5_apex[ o+p ];
        o += a;
        a = a*5u;
        p *= 5u;
        if( val.x <= key ) {
            p++;
            key -=val.x;
            if( val.y <= key ) {
                p++;
                key-=val.y;
                if( val.z <= key ) {
                    p++;
                    key-=val.z;
                    if( val.w <= key ) {
                        p++;
                        key-=val.w;
                    }
                }
            }
        }
    }
#endif
    for(; l<CUHPMC_LEVELS; l++) {
        uvec4 val = texelFetch( hp5_tex, int(hp5_offsets[l]+p) );
        p *= 5;
        if( val.x <= key ) {
            p++;
            key -=val.x;
            if( val.y <= key ) {
                p++;
                key-=val.y;
                if( val.z <= key ) {
                    p++;
                    key-=val.z;
                    if( val.w <= key ) {
                        p++;
                        key-=val.w;
                    }
                }
            }
        }
    }
    pos = p;
    key_remainder = key;
}
