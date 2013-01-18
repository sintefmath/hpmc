#version 150
/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
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
in vec2 vbo_texcoord;
in vec3 vbo_normal;
in vec3 vbo_position;

// input from interleaved GL_T2F_N3F_V3F buffer
// pass output to GS, position in gl_Position
out vec3 invel;
out vec2 ininfo;
out vec3 inpos;

void
main()
{
    invel       = vbo_normal;
    ininfo      = vbo_texcoord;
    inpos       = vbo_position;
    //gl_Position = vec4(vbo_position,1.0);
};
