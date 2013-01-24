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
out vec3                normal;

layout(binding=0)   uniform usampler2D      intersection_table_tex;
layout(binding=1)   uniform samplerBuffer   field_tex;
layout(binding=2)   uniform usamplerBuffer  hp5_tex;
layout(binding=3)   uniform usamplerBuffer  cases_tex;
                    uniform float           iso;
                    uniform uint            hp5_offsets[ CUHPMC_LEVELS ];
                    uniform mat4            modelviewprojection;
                    uniform mat3            normalmatrix;




void
main()
{
    uint key = gl_VertexID;
    uint pos = 0;

    // Use bindable uniform (=constmem) for apex. Layout is known here, so we
    // can directly calculate the offsets.
    uint l = 0;
    for(; l<CUHPMC_LEVELS; l++) {
        uvec4 val = texelFetch( hp5_tex, int(hp5_offsets[l]+pos) );
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

    uint key_remainder = key;

    // Calc 3D grid pos from linear input stream pos.
    uint c_lix = pos / 800;
    uint t_lix = pos % 800;
    uvec3 ci = uvec3( 31u*( c_lix % CUHPMC_CHUNKS_X ),
                       5u*( (c_lix/CUHPMC_CHUNKS_X) % CUHPMC_CHUNKS_Y ),
                       5u*( (c_lix/CUHPMC_CHUNKS_X) / CUHPMC_CHUNKS_Y ) );
    uvec3 i0 = uvec3( ci.x + ((t_lix / 5u)%32u),
                      ci.y + ((t_lix / 5u)/32u),
                      ci.z + ( t_lix%5u ) );

    uint mc_case = texelFetch( cases_tex, int(pos) ).r;

    uint isec = texelFetch( intersection_table_tex,
                           ivec2( key_remainder, mc_case ), 0 ).r;
    uvec3 oa = i0 + uvec3( (isec   )&1u,
                           (isec>>1u)&1u,
                           (isec>>2u)&1u );
    uvec3 ob = i0 + uvec3( (isec>>3u)&1u,
                           (isec>>4u)&1u,
                           (isec>>5u)&1u );
    uint ixa = oa.x + oa.y*CUHPMC_FIELD_ROW_PITCH + oa.z*CUHPMC_FIELD_SLICE_PITCH;
    uint ixb = ob.x + ob.y*CUHPMC_FIELD_ROW_PITCH + ob.z*CUHPMC_FIELD_SLICE_PITCH;


    // Sample edge points and determine gradients using discrete differentials.

    float fa = texelFetch( field_tex, int(ixa) ).r;

    vec3 ga = vec3( texelFetch( field_tex, int(ixa + 1                      )   ).r -fa,
                    texelFetch( field_tex, int(ixa + CUHPMC_FIELD_ROW_PITCH )   ).r -fa,
                    texelFetch( field_tex, int(ixa + CUHPMC_FIELD_SLICE_PITCH ) ).r -fa );

    float fb = texelFetch( field_tex, int(ixb) ).r;
    vec3 gb = vec3( texelFetch( field_tex, int( ixb + 1                        ) ).r -fb,
                    texelFetch( field_tex, int( ixb + CUHPMC_FIELD_ROW_PITCH   ) ).r -fb,
                    texelFetch( field_tex, int( ixb + CUHPMC_FIELD_SLICE_PITCH ) ).r -fb );

    // Find zero-crossing of linear polynomial and emit vertex.
    float t = (iso-fa)/(fb-fa);
    float omt = 1.0f-t;
    normal = normalmatrix * mix(ga,gb,t);
    const vec3 scale = vec3( CUHPMC_SCALE_X, CUHPMC_SCALE_Y, CUHPMC_SCALE_Z );

    vec3 P = scale*mix( vec3(oa), vec3(ob), t );
    gl_Position = modelviewprojection * vec4( P, 1.0 );
}
