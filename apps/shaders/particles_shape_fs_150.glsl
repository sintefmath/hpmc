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
varying in vec3 normal_cs;
varying in float grad_length;
void
main()
{
    const vec3 v = vec3(0.0, 0.0, 1.0 );
    vec3 l = normalize(vec3(1.0, 1.0, 1.0));
    vec3 h = normalize( v + l );
    vec3 cn = normalize( normal_cs );
    float diff = max(0.0,dot( cn, l ) )
               + max(0.0,dot(-cn, l ) );
    float spec = pow( max( 0.0, dot( cn, h) ), 30.0 )
               + pow( max( 0.0, dot(-cn, h) ), 30.0 );
    gl_FragColor = vec4( 0.1, 0.2, 0.7, 0.0) * diff
                 + vec4( 1.0, 1.0, 1.0, 0.0) * spec;
};
