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
layout(triangles) in;
layout(points, max_vertices=1) out;

// geometry shader is run once per triangle and emits one or nil points

// an offset we use to randomize which primitive that generates points
uniform int offset;

// governs how likely it is that a triangle will produce a point
uniform int threshold;
uniform mat4 P;

in vec3 normal[3];
in vec3 position[3];

// varyings that will be recorded
out vec2 info;
out vec3 vel;
out vec3 pos;

void
main()
{
    if( int(offset + gl_PrimitiveIDIn) % threshold == 0 ) {
        int side = (gl_PrimitiveIDIn / threshold) %2;
        info = vec2( 1.0, 1.0 );
//       position new particle on center of triangle
        pos = (1.0/3.0)*( position[0].xyz +
                          position[1].xyz +
                          position[2].xyz )
//       and push it slightly off the surface along the normal direction
            + (side!=0?0.02:-0.02)*normalize( normal[0] +
                                              normal[1] +
                                              normal[2] );
//       initial velocity is zero
        vel = vec3(0.0);
        gl_Position = P * vec4(pos, 1.0);
        EmitVertex();
    }
};
