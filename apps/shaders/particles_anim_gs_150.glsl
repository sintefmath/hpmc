#version 150
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
layout(points) in;
layout(points, max_vertices=1) out;

// input passed from VS
in vec3 invel[];
in vec2 ininfo[];
in vec3 inpos[];

// output varyings that get fed back
out vec3 pos;
out vec3 vel;
out vec2 info;

// timestep
uniform float dt;
uniform float iso;
uniform mat4 P;
uniform mat4 MV;
uniform mat4 MV_inv;
uniform mat3 NM;

float
HPMC_fetch( vec3 p );

vec4
HPMC_fetchGrad( vec3 p );

void
main()
{
    info = ininfo[0] - vec2( 0.1*dt, dt );
    vec3 vel_a_c = invel[0];
    vec3 pos_a_c = inpos[0];
    vec3 acc_b_c = vec3( 0.0, -0.6, 0.0 );
    vec3 vel_b_c = vec3(0.f);
    vec3 pos_b_c = vec3(0.f);
    const int steps = 32;
    float sdt = (1.0/float(steps))*dt;

    // object space pos of a
    vec4 pos_a_ho = MV_inv * vec4( pos_a_c, 1.0 );
    vec3 pos_a_o = (1.0/pos_a_ho.w)*pos_a_ho.xyz;

    for( int s=0; s<steps; s++ ) {
//       integrate
        vel_b_c = vel_a_c + sdt * acc_b_c;
        pos_b_c = pos_a_c + sdt * vel_b_c;
//       calc object space pos of b
        vec4 pos_b_ho = MV_inv * vec4( pos_b_c, 1.0 );
        vec3 pos_b_o = (1.0/pos_b_ho.w)*pos_b_ho.xyz;
//       surface interaction only happen inside object space unit cube
        if( all( lessThan( abs(pos_b_o-vec3(0.5)), vec3(0.5) ) ) ) {
//           First, find the direction towards the surface
            vec4 gradsample_a = HPMC_fetchGrad( pos_a_o )-vec4(0.0,0.0,0.0,iso);
            vec3 to_surf_o = -0.01*sign(gradsample_a.w)*normalize(gradsample_a.xyz);
            vec3 to_surf_c = NM * to_surf_o;
//           Check if particle is moving towards the surface
            if( dot(vel_b_c, to_surf_c) > 0.0 ) {
//              Then, check the scalar feld a small step towards the surface
                vec3 to_surf_pos = pos_a_o + to_surf_o;
                float to_surf_field = HPMC_fetch( to_surf_pos )-iso;
//               If field changes sign, move particle slighly backwards
//               to minimize the risk of the particle falling through
//               the surface. Note that just checking signs might have
//               problems at multiple zeros, which happen with these
//               algebraic surfaces.
                if( (to_surf_field)*(gradsample_a.w) <= 0.0 ) {
//                   Determine closest point on surface
                    float t = -gradsample_a.w/(to_surf_field-gradsample_a.w);
//                   And move the particle a small step backwards
                    pos_a_o = mix( pos_a_o, to_surf_pos, t ) - to_surf_o;
//                   Update camera-space position,
                    vec4 pos_a_hc = MV * vec4( pos_a_o, 1.0 );
                    vec3 new_pos_a_c = (1.0/pos_a_hc.w)*pos_a_hc.xyz;
//                   Find direction we pushed in camera-space
                    vec3 to_surf_n_c = normalize( to_surf_c );
//                   Determine amount of movement towards the surface
                    float to_surf_vel = dot( vel_b_c, to_surf_n_c );
//                   Kill velocity into the surface
                    vel_b_c -= to_surf_vel*to_surf_n_c;
//                   update position of a
                    pos_a_c = new_pos_a_c;
                    pos_b_c = pos_a_c + sdt * vel_b_c;
                    vec4 pos_b_ho = MV_inv * vec4( pos_b_c, 1.0 );
                    pos_b_o = (1.0/pos_b_ho.w)*pos_b_ho.xyz;
                    info.y = 1.0;
                }
            }
            float field_a = HPMC_fetch( pos_a_o ) - iso;
            float field_b = HPMC_fetch( pos_b_o ) - iso;

            // check for sign change
            if( field_a*field_b <= 0.0 ) {

                // determine zero-crossing
                float t = -field_a/(field_b-field_a);

                // step back to intersection
                pos_b_c -= (1.0-t)*dt*vel_b_c;

                // point of intersection in object space
                vec3 pos_i_o = mix( pos_a_o, pos_b_o, t );

                // gradient at intersection used to get surface normal
                vec3 nrm_i_c = normalize( NM * HPMC_fetchGrad( pos_i_o ).xyz );

                // reflect velocity
                vel_b_c = reflect( vel_b_c, nrm_i_c );

                // step rest of timestep in reflected direction
                pos_b_c += (1.0-t)*dt*vel_b_c;
                vel_b_c *= 0.98;
                info.y = 1.0;
            }
        }
        vel_a_c = vel_b_c;
        pos_a_c = pos_b_c;
        pos_a_o = pos_b_o;
    }
    vel = vel_b_c;
    pos = pos_b_c;
    vec4 hpos = P * vec4(pos, 1.0);
    vec3 norm = (1.0/hpos.w)*hpos.xyz;

    //   only emit particles inside the frustum and that are not too old
    if( (info.x > 0.0) && all( lessThan( abs(norm), vec3(1.0) ) ) ) {
        gl_Position = hpos;
        EmitVertex();
    }
};
