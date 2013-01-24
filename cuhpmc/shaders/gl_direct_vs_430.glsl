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

layout(binding=0)   uniform isampler2D      intersection_table_tex;
layout(binding=1)   uniform samplerBuffer   field_tex;
layout(binding=2)   uniform usamplerBuffer  hp5_tex;
                    uniform float           iso;
                    uniform uint            hp5_offsets[ CUHPMC_LEVELS ];
                    uniform mat4            modelviewprojection;
                    uniform mat3            normalmatrix;

#ifdef USE_BINDABLE_UNIFORM
layout(std140)      uniform HP5Apex {
                        ivec4               hp5_apex[528];
                    };
#endif

#ifdef CASE_BUFFER_LOAD
uniform float* cases;
#else
layout(binding=3)   uniform usamplerBuffer  cases_tex;
#endif


void
main()
{
    int key = gl_VertexID;
    int pos = 0;

    // Use bindable uniform (=constmem) for apex. Layout is known here, so we
    // can directly calculate the offsets.
    int o = 1;
    int a = 1;
    int l = 0;
#ifdef USE_BINDABLE_UNIFORM
    for(; l<4; l++) {
        ivec4 val = hp5_apex[ o+pos ];
        o += a;
        a = a*5;
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

#endif
    for(; l<CUHPMC_LEVELS; l++) {
        ivec4 val = ivec4(texelFetch( hp5_tex, int(hp5_offsets[l]+pos) ));
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


    int key_remainder = key;

    // Calc 3D grid pos from linear input stream pos.
    int c_lix = pos / 800;
    int t_lix = pos % 800;
    ivec3 ci = ivec3( 31*( c_lix % CUHPMC_CHUNKS_X ),
                      5*( (c_lix/CUHPMC_CHUNKS_X) % CUHPMC_CHUNKS_Y ),
                      5*( (c_lix/CUHPMC_CHUNKS_X) / CUHPMC_CHUNKS_Y ) );
    ivec3 i0 = ivec3( ci.x + ((t_lix / 5)%32),
                      ci.y + ((t_lix / 5)/32),
                      ci.z + ( t_lix%5 ) );

#ifdef CASE_BUFFER_LOAD
    int mc_case = 0;
#else


#endif

    // Determine intersection edge end-points

#ifdef CASE_BUFFER_PACKED
    int ix0 = i0.x +
              i0.y*CUHPMC_FIELD_ROW_PITCH +
              i0.z*CUHPMC_FIELD_SLICE_PITCH;
    int mc_case = int(texelFetch( cases_tex, ix0 ).a);
    int isec = int( texelFetch( intersection_table_tex, ivec2( key_remainder, mc_case ), 0 ).a );
    int ixa = ix0;
    int ixb = ix0;
    ivec3 oa = i0;
    ivec3 ob = i0;
    if( ((isec)&1)  != 0 ) {
        ixa += 1;
        oa.x += 1;
    }
    if( ((isec>>1)&1) != 0 ) {
        ixa += CUHPMC_FIELD_ROW_PITCH;
        oa.y += 1;
    }
    if( ((isec>>2)&1) != 0  ) {
        ixa += CUHPMC_FIELD_SLICE_PITCH;
        oa.z += 1;
    }
    if( ((isec>>3)&1)  != 0 ) {
        ixb += 1;
        ob.x += 1;
    }
    if( ((isec>>4)&1)  != 0 ) {
        ixb += CUHPMC_FIELD_ROW_PITCH;
        ob.y += 1;
    }
    if( ((isec>>5)&1) != 0  ) {
        ixb += CUHPMC_FIELD_SLICE_PITCH;
        ob.z += 1;
    }
#else
    int mc_case = int(texelFetch( cases_tex, pos ).a);
    int isec = int( texelFetch( intersection_table_tex,
                           ivec2( key_remainder, mc_case ), 0 ).a );
    ivec3 oa = i0 + ivec3( (isec   )&1,
                           (isec>>1)&1,
                           (isec>>2)&1 );
    ivec3 ob = i0 + ivec3( (isec>>3)&1,
                           (isec>>4)&1,
                           (isec>>5)&1 );
    int ixa = oa.x + oa.y*CUHPMC_FIELD_ROW_PITCH + oa.z*CUHPMC_FIELD_SLICE_PITCH;
    int ixb = ob.x + ob.y*CUHPMC_FIELD_ROW_PITCH + ob.z*CUHPMC_FIELD_SLICE_PITCH;
#endif


    // Sample edge points and determine gradients using discrete differentials.

    float fa = texelFetch( field_tex, ixa ).a;

    vec3 ga = vec3( texelFetch( field_tex, ixa + 1           ).a -fa,
                    texelFetch( field_tex, ixa + CUHPMC_FIELD_ROW_PITCH   ).a -fa,
                    texelFetch( field_tex, ixa + CUHPMC_FIELD_SLICE_PITCH ).a -fa );

    float fb = texelFetch( field_tex, ixb ).a;
    vec3 gb = vec3( texelFetch( field_tex, ixb + 1           ).a -fb,
                    texelFetch( field_tex, ixb + CUHPMC_FIELD_ROW_PITCH   ).a -fb,
                    texelFetch( field_tex, ixb + CUHPMC_FIELD_SLICE_PITCH ).a -fb );

    // Find zero-crossing of linear polynomial and emit vertex.
    float t = 0.5f;//(iso-fa)/(fb-fa);
    float omt = 1.0f-t;
    normal = normalmatrix * (omt*ga + t*gb);
    vec3 P = vec3( CUHPMC_SCALE_X, CUHPMC_SCALE_Y, CUHPMC_SCALE_Z )*vec3( i0 );

    gl_Position = modelviewprojection * vec4( P, 1.0 );
    /*gl_Position = modelviewprojection * vec4( omt*vec3(oa) +
                                              t*vec3(ob),
                                              1.0 );*/
    //gl_Position = modelviewprojection * vec4( 0.5, 0.5, 0.5, 1.0 );
}
