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
layout(triangle_strip, max_vertices=4) out;

// input passed from VS
in vec3 invel[];
in vec2 ininfo[];
in vec3 inpos[];

out vec2 tp;
out float depth;

uniform mat4 P;

void
main()
{
    float i = ininfo[0].x;

    // determine size of billboard
    float r = 0.005 + 0.005*max(0.0,pow(i,30.0));

    // color of particle
    gl_FrontColor.xyz = vec3( pow(i,30.0), ininfo[0].y, 0.8 );
    vec4 p = vec4(inpos[0], 1.0);

    // calculate depth, see note in fragment shader
    vec4 ppp = (P * p);
    depth = 0.5*((ppp.z)/ppp.w)+0.5;

    // emit a billboard quad as a tiny triangle strip
    tp = vec2(-1.0,-1.0);
    gl_Position = P*(p + vec4(-r,-r, 0.0, 1.0 ));
    EmitVertex();

    tp = vec2(-1.0, 1.0);
    gl_Position = P*(p + vec4(-r, r, 0.0, 1.0 ));
    EmitVertex();

    tp = vec2( 1.0,-1.0);
    gl_Position = P*(p + vec4( r,-r, 0.0, 1.0 ));
    EmitVertex();

    tp = vec2( 1.0, 1.0);
    gl_Position = P*(p + vec4( r, r, 0.0, 1.0 ));
    EmitVertex();
}
