#version 430
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

in GO {
    vec3                normal;
} in_f;

layout(location=0)   out vec4 fragment;

void
main()
{
    vec3 v = vec3( 0.0, 0.0, 1.0 );
    vec3 n = normalize( in_f.normal );
    if( n.z < 0.0 ) {
        n = -n;
    }
    vec3 r = reflect( v, n );
    vec3 h = 0.5*(v+n);
    vec3 c_r = vec3(0.4, 1.3, 2.0) * max( 0.0, -r.y )
             + vec3(0.5, 0.4, 0.2) * pow( max( 0.0, r.y), 3.0 );
    vec3 c_s = vec3(0.7, 0.9, 1.0) * pow( max( 0.0, dot( v, h ) ), 50.0 );
    vec3 c_f = vec3(0.8, 0.9, 1.0) * pow( 1.0-abs(n.z), 5.0 );
    fragment = vec4( c_r + c_s + c_f, 0.2 );
}
