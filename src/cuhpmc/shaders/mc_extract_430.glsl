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

layout(binding=0)   uniform usampler2D      intersection_table_tex;
layout(binding=1)   uniform samplerBuffer   field_tex;
                    uniform float           iso;

void
mc_extract( out vec3 P, out vec3 N, in uvec3 i0, in uint mc_case, in uint remainder )
{
    uint isec = texelFetch( intersection_table_tex,
                           ivec2( remainder, mc_case ), 0 ).r;
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

    const vec3 scale = vec3( CUHPMC_SCALE_X, CUHPMC_SCALE_Y, CUHPMC_SCALE_Z );

    P = scale*mix( vec3(oa), vec3(ob), t );
    N = mix(ga,gb,t);

}
