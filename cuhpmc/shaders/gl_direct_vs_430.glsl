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

uniform mat4            modelviewprojection;
uniform mat3            normalmatrix;

void
hp5_downtraverse( out uint pos, out uint key_remainder, in uint key );

void
mc_extract( out vec3 P, out vec3 N, in uint pos, in uint remainder );

void
main()
{
    uint pos;
    uint key_remainder;
    hp5_downtraverse( pos, key_remainder, gl_VertexID );
    vec3 P;
    vec3 N;
    mc_extract( P, N, pos, key_remainder );

    normal = normalmatrix * N;
    gl_Position = modelviewprojection * vec4( P, 1.0 );
}
