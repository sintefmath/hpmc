#version 420
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
layout(location=0)  in      vec3 vbo_pos;
layout(location=1)  in      vec3 vbo_nrm;
                    out     vec3 normal;
                    uniform mat4 PM;
                    uniform mat3 NM;

void
main()
{
    gl_Position = PM * vec4( vbo_pos, 1.0 );
    normal = NM * vbo_nrm;
}

