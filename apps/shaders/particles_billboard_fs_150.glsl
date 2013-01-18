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
in vec2 tp;
in float depth;

out vec4 fragment;

uniform vec3 color;

void
main()
{
    fragment = pow((max(1.0-length(tp),0.0)),2.0)*vec4(color,1.0);
// for some reason the depth test doesn't work as expected if the depth
// isn't written... at least on my setup.
    gl_FragDepth = depth;
}
